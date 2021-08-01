class ProcgenConfig:
    """
    configuration based on:

    Machado, M. C., et al. (2018). "Revisiting the arcade learning environment: Evaluation protocols and open
    problems for general agents." Journal of artificial intelligence research 61: 523-562.
    """

    def __init__(self):
        self.debug = True  # whether to print state and observation images to tensorboard
        self.display = True  # display game and max q plot
        self.double_dqn = False
        # self.device = 'cuda:0'  # what device to do learning updates with

        # NETWORK ARCHITECTURE
        self.qnet = 'proc_mixer'

        # ENVIRONMENT TYPE
        self.env_type = 'procgen'  # 'atari' or 'procgen'

        # PROCGEN ENVIRONMENT DEFAULTS
        self.game = 'bigfish'
        self.proc_num_levels = 0
        self.proc_start_level = 0
        self.proc_paint_vel_info = False
        self.use_generated_assets = False
        self.proc_debug = False
        self.proc_center_agent = True
        self.proc_use_sequential_levels = False
        self.proc_distribution_mode = "easy"
        self.proc_use_backgrounds = True
        self.proc_restrict_themes = False
        self.proc_use_monochrome_assets = False

        self.max_episode_length = int(1e9)

        # OPTIMIZER
        self.optimizer = 'adam'  # 'adam', 'rms_prop'

        # ADAM SETTINGS
        self.adam_lr = 0.0000625
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 0.00015
        self.adam_weight_decay = 0

        # RMS PROP SETTINGS
        self.rms_prop_lr = 0.00025
        self.rms_prop_alpha = 0.95
        self.rms_prop_eps = 0.1 / 32.0
        self.rms_prop_weight_decay = 0
        self.rms_prop_momentum = 0

        # LOGGING AND SAVING
        self.log_inside_qnet = True
        self.log_inside_qnet_freq = 10_000
        self.logging_freq = 10_000  # steps
        self.save_param_freq = 1_000_000

        # REPLAY BUFFER
        self.buffer_size = 1_000_000
        self.frame_history_len = 4
        self.batch_size = 32

        # TRAINING
        self.nsteps_train = 8_000_000
        self.learning_start = 50_000  # 50_000
        self.learning_freq = 4
        self.gamma = 0.99
        self.target_update_freq = 10_000

        # EPSILON
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_steps = 1_000_000
