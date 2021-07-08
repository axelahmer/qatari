class TestConfig:
    """
    configuration used for testing purposes.
    """
    def __init__(self):
        self.debug = True  # whether to print state and observation images to tensorboard
        self.display = True  # display game and max q plot
        # self.device = 'cuda:0'  # what device to do learning updates with

        # NETWORK ARCHITECTURE
        self.qnet = 'split1'  # 'nature', 'summer', 'mixer'

        # ENVIRONMENT
        self.game = 'pong'
        self.seed = 123
        self.mode = 0
        self.difficulty = 0
        self.frame_skip = 4
        self.repeat_action_probability = 0.25
        self.full_action_space = True
        self.noop_max = 0
        self.terminal_on_life_loss = True
        self.max_episode_length = 18000 // self.frame_skip # i.e. 18000 frames

        # OPTIMIZER
        self.optimizer = 'adam'  # 'adam', 'rms_prop'

        self.adam_lr = 0.0000625
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 0.00015
        self.adam_weight_decay = 0

        self.rms_prop_lr = 0.00025
        self.rms_prop_alpha = 0.95
        self.rms_prop_eps = 0.1 / 32.0
        self.rms_prop_weight_decay = 0
        self.rms_prop_momentum = 0

        # LOGGING AND SAVING
        self.logging_freq = 5_000  # steps
        self.save_param_freq = 500_000

        # REPLAY BUFFER
        self.buffer_size = 1_000_000  # 1_000_000
        self.frame_history_len = 4
        self.batch_size = 32

        # TRAINING
        self.nsteps_train = 10_000_000  # 200_000_000
        self.learning_start = 5_000  # 50_000
        self.learning_freq = 4
        self.gamma = 0.99
        self.target_update_freq = 10_000

        # EPSILON
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_steps = 1_000_000
