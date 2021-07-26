import sys
from core.q_learner import QLearner
import torch
import numpy as np
from modules import module_dict
import torch.nn as nn


class DQNLearner(QLearner):
    def __init__(self, config=None):
        super().__init__(config=config)

        self.logged_graph = False

        if hasattr(config, 'device'):
            self.device = config.device
        else:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # build networks
        in_channels = self.env.observation_space.shape[-1] * self.config.frame_history_len
        out_channels = self.env.action_space.n

        self.network_ptr = module_dict[self.config.qnet]
        self.q_network = self.network_ptr(in_channels, out_channels, self.writer).to(self.device)
        self.target_network = self.network_ptr(in_channels, out_channels, self.writer).to(self.device)

        # copy q network params to target network
        self.update_target_params()

        # add optimizer
        self.optimizer = self.make_optimizer()

        # loss fn
        self.loss_fn = nn.HuberLoss()

        print(f'successfully built model on device: {self.device}\nconfig: {self.config.__dict__}')

    def get_greedy_action(self, state, t):

        s = torch.tensor(state, dtype=torch.uint8, device=self.device).unsqueeze(0)
        # process state to be [0,1] floats
        s = self.process_state(s)
        # forward pass of q network
        if self.config.log_inside_qnet is True and t % self.config.log_inside_qnet_freq == 0:
            q_values = self.q_network.log_forward(s)
        else:
            with torch.no_grad():
                q_values = self.q_network.forward(s)

        if self.render_qnet:
            self.q_network.render()

        q_values = q_values.squeeze().to('cpu').tolist()
        action = np.argmax(q_values)
        q_max = q_values[action]

        return action, q_max

    def train_step(self, t, replay_buffer):
        # executed every training step after environment interaction
        loss = None

        # update q network
        if t >= self.config.learning_start and t % self.config.learning_freq == 0:
            loss = self.learn(t, replay_buffer)

        # update target network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()

        # TODO: Implement saving weights and other logs

        return loss

    def learn(self, t, replay_buffer):
        # sample minibatch from replay buffer
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(self.config.batch_size)

        # convert to tensor and move to correct device
        s_batch = torch.tensor(s_batch, dtype=torch.uint8, device=self.device)
        a_batch = torch.tensor(a_batch, dtype=torch.int64, device=self.device)
        r_batch = torch.tensor(r_batch, dtype=torch.float32, device=self.device)
        sp_batch = torch.tensor(sp_batch, dtype=torch.uint8, device=self.device)
        done_mask_batch = torch.tensor(done_mask_batch, dtype=torch.bool, device=self.device)

        # clip rewards
        r_batch.clip_(min=-1., max=1.)

        # process state batches on device
        s0 = self.process_state(s_batch)
        s1 = self.process_state(sp_batch)

        # calc qs and q targets
        q0 = self.q_network(s0).gather(1, a_batch.unsqueeze(1)).squeeze()
        with torch.no_grad():
            if self.config.double_dqn is True:
                # double dqn q target
                _, a1 = self.q_network(s1).max(1)
                q1_bootstrap = self.target_network(s1).gather(1, a1.unsqueeze(1)).squeeze()
            else:
                # standard dqn q target
                q1_bootstrap, _ = self.target_network(s1).max(1)

        td_targets = torch.where(done_mask_batch,
                                 r_batch,
                                 r_batch + q1_bootstrap * self.config.gamma)

        # huber loss
        loss = self.loss_fn(q0, td_targets.detach())

        # optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log graph if not already done
        if self.logged_graph is False:
            self.writer.add_graph(self.q_network, s0)
            self.logged_graph = True

        return loss.item()

    def update_target_params(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    @staticmethod
    def process_state(state: torch.Tensor):
        # scale state and arrange axes for input to networks
        return (state / 255.0).permute(0, 3, 1, 2)

    def make_optimizer(self):
        if self.config.optimizer == 'adam':
            return torch.optim.Adam(params=self.q_network.parameters(),
                                    lr=self.config.adam_lr,
                                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                                    eps=self.config.adam_eps,
                                    weight_decay=self.config.adam_weight_decay)
        elif self.config.optimizer == 'rms_prop':
            return torch.optim.RMSprop(params=self.q_network.parameters(),
                                       lr=self.config.rms_prop_lr,
                                       alpha=self.config.rms_prop_alpha,
                                       eps=self.config.rms_prop_eps,
                                       weight_decay=self.config.rms_prop_weight_decay,
                                       momentum=self.config.rms_prop_momentum,
                                       centered=True)
        else:
            print('optimizer incorrectly specified')
            sys.exit()


if __name__ == '__main__':
    # build model
    model = DQNLearner()

    # run model
    model.run()
