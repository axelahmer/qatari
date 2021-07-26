# Class outline inspired by Stanford's DQN Assignment

from collections import deque
import numpy as np
from utils.linear_schedule import LinearSchedule
from utils.replay_buffer import ReplayBuffer
from gym.envs.atari.atari_env import AtariEnv
from utils.preprocess_wrapper import AtariPreprocessing
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from configs.machado import MachadoConfig


class QLearner:

    def __init__(self, config=None):

        if config is None:
            self.config = MachadoConfig()
            print('Using default Machado Configuration')
        else:
            self.config = config

        self.env = self.make_env()
        self.q_network = None
        self.render_train = True
        self.render_qnet = False

        # logging stuff
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.PATH = f'results/{self.config.env_type}/{self.config.game}/{self.config.qnet}/{self.config.seed}/{timestamp}'
        self.MODEL_PATH = self.PATH + 'model.params'

        self.writer = SummaryWriter(log_dir=self.PATH, max_queue=100)
        self.writer.add_text('config_info', str(self.config.__dict__))

    # IMPLEMENT ME
    def get_greedy_action(self, q_input, t):
        raise NotImplementedError

    # IMPLEMENT ME
    def train_step(self, t, replay_buffer):
        raise NotImplementedError

    def make_env(self):
        # create environment following Machado et al. 2018
        if self.config.env_type == 'atari':
            env = AtariEnv(
                game=self.config.game,
                mode=self.config.mode,
                difficulty=self.config.difficulty,
                obs_type='image',
                frameskip=1,  # skipping handled by wrapper
                repeat_action_probability=self.config.repeat_action_probability,
                full_action_space=self.config.full_action_space
            )

            env = AtariPreprocessing(
                env=env,
                noop_max=self.config.noop_max,
                frame_skip=self.config.frame_skip,
                screen_size=84,
                terminal_on_life_loss=self.config.terminal_on_life_loss,
                grayscale_obs=True,
                grayscale_newaxis=True,  # extra axis required for replay buffer
                scale_obs=False
            )

            # set seed
            env.seed(self.config.seed)

        elif self.config.env_type == 'procgen':
            env = None
        else:
            raise NotImplementedError

        return env

    def train(self):

        # initialize replay buffer and schedules
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.frame_history_len)
        eps_schedule = LinearSchedule(self.config.eps_start, self.config.eps_end, self.config.eps_steps)

        # things we want to log
        log_recent_losses = deque(maxlen=100)
        log_recent_max_qs = deque(maxlen=1_000)
        log_recent_clipped_rewards = deque(maxlen=1_000)
        log_recent_episode_scores = deque(maxlen=100)
        log_recent_episode_scores_clipped = deque(maxlen=100)
        if self.config.terminal_on_life_loss is False:
            log_recent_life_scores = deque(maxlen=100)
            log_recent_life_scores_clipped = deque(maxlen=100)

        for _ in range(1000):
            log_recent_max_qs.append(0)

        # add key listener to render window
        if self.config.display:
            self.env.reset()
            self.add_key_listener()

            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 2))
            line_maxq, = ax.plot(range(1000), log_recent_max_qs, '-', alpha=0.8)
            ax.set_title('max q')

        # main training loop:
        t = 0
        episode = 0
        frame = self.env.reset()
        while t < self.config.nsteps_train:
            t_episode = 0
            episode += 1
            episode_score = 0.
            episode_score_clipped = 0.
            if self.config.terminal_on_life_loss is False:
                life_score = 0.
                life_score_clipped = 0.
            while True:
                t += 1
                t_episode += 1

                # replay memory stuff
                idx = replay_buffer.store_frame(frame)
                state = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                greedy_action, max_q = self.get_greedy_action(state, t)
                action = self.get_eps_greedy_action(greedy_action, eps_schedule.epsilon)

                # store max q value
                log_recent_max_qs.append(max_q)

                # perform action in env and record reward
                new_frame, reward, done, info = self.env.step(action)

                # enforce game termination after specified num frames
                if t_episode >= self.config.max_episode_length:
                    done = True

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                frame = new_frame

                # perform a training step and record loss
                loss = self.train_step(t, replay_buffer)  # lr_schedule.epsilon
                if loss is not None:
                    log_recent_losses.append(loss)

                # update schedulers
                eps_schedule.update(t)

                # clip reward
                reward_clipped = np.clip(reward, -1., 1.)

                # update score logs
                episode_score += reward
                episode_score_clipped += reward_clipped
                log_recent_clipped_rewards.append(reward_clipped)
                if self.config.terminal_on_life_loss is False:
                    life_score += reward
                    life_score_clipped += reward_clipped
                    if info['life_loss']:
                        log_recent_life_scores.append(life_score)
                        log_recent_life_scores_clipped.append(life_score_clipped)
                        life_score = 0.
                        life_score_clipped = 0.

                # logging
                if t >= self.config.learning_start and t % self.config.logging_freq == 0:
                    self.writer.add_scalar('avg_maxq_(1000_timesteps)', np.mean(log_recent_max_qs), t)
                    self.writer.add_scalar('avg_reward_clipped_(1000_timesteps)', np.mean(log_recent_clipped_rewards),
                                           t)
                    self.writer.add_scalar('avg_episode_score_(100_episodes)', np.mean(log_recent_episode_scores), t)
                    self.writer.add_scalar('avg_episode_score_clipped_(100_episodes)',
                                           np.mean(log_recent_episode_scores_clipped), t)
                    self.writer.add_scalar('avg_losses_(100_updates)', np.mean(log_recent_losses), t)
                    if self.config.terminal_on_life_loss is False:
                        self.writer.add_scalar('avg_life_score_(100_lives)', np.mean(log_recent_life_scores), t)
                        self.writer.add_scalar('avg_life_score_clipped_(100_lives)',
                                               np.mean(log_recent_life_scores_clipped), t)
                    if self.config.debug:
                        self.writer.add_image('debug_frame', frame, dataformats='HWC')
                        self.writer.add_images('debug_state', np.expand_dims(state, 0), dataformats='CHWN')

                # update display
                if self.config.display:
                    self.env.viewer.window.dispatch_events()
                    if self.render_train:
                        # render game
                        self.env.render()
                        # plot running max q
                        line_maxq.set_ydata(log_recent_max_qs)
                        ax.relim()
                        ax.autoscale_view()  # automatic axis scaling
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)

                # save params
                if t % self.config.save_param_freq == 0:
                    torch.save(self.q_network.state_dict(), self.MODEL_PATH)
                    print(f'weights saved at t={t}')

                # reset env if game over
                if info['game_over']:
                    frame = self.env.reset()

                # end episode
                if done:
                    break

            # store episode scores
            log_recent_episode_scores.append(episode_score)
            log_recent_episode_scores_clipped.append(episode_score_clipped)

            # report episode stats
            # self.writer.add_scalar('score', episode_score, t)
            # self.writer.add_scalar('score_clipped', episode_score_clipped, t)

            # print episode info to console
            print(f'| episode: {str(episode).rjust(4, " ")} '
                  f'| steps: {str(t_episode).rjust(4, " ")} '
                  f'| score: {episode_score:+.2f} '
                  f'| clipped score: {episode_score_clipped:+.2f} '
                  f'| eps: {eps_schedule.epsilon:.2f} '
                  f'| t: {t:,} |')

    def load(self):
        self.q_network.load_state_dict(torch.load(self.MODEL_PATH))

    def run(self):
        self.train()

    def get_eps_greedy_action(self, greedy_action, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return greedy_action

    def add_key_listener(self):
        from pyglet.window import key
        self.env.render()

        # allow ability to display live training with <spacebar> on render window
        def on_key_press(symbol, modifiers):
            if symbol == key.SPACE:
                self.render_train = not self.render_train
            if symbol == key.N:
                self.render_qnet = not self.render_qnet

        self.env.viewer.window.push_handlers(on_key_press)
