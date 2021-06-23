class DefaultConfig:
    debug = True  # whether to print state and observation images to tensorboard
    display = True  # display game and max q plot
    # device = 'cuda:0'  # what device to do learning updates with

    # ENVIRONMENT
    seed = 123
    mode = 0
    difficulty = 0
    frame_skip = 4
    repeat_action_probability = 0.25
    full_action_space = True
    noop_max = 0
    terminal_on_life_loss = False

    # OPTIMIZER
    optimizer = 'adam'  # 'adam', 'rms_prop'

    adam_lr = 0.0000625
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 0.00015
    adam_weight_decay = 0

    rms_prop_lr = 0.00025
    rms_prop_alpha = 0.95
    rms_prop_eps = 0.1 / 32.0

    # LOGGING AND SAVING
    logging_freq = 1000  # steps
    save_param_freq = 500_000

    # REPLAY BUFFER
    buffer_size = 1_000_000  # 1_000_000
    frame_history_len = 4
    batch_size = 32

    # TRAINING
    nsteps_train = 200_000_000
    learning_start = 5_000  # 50_000
    learning_freq = 4
    gamma = 0.99
    target_update_freq = 10_000

    # EPSILON
    eps_start = 1.0
    eps_end = 0.01
    eps_steps = 1_000_000

    # LEARNING RATE
    # lr_start = 0.1
    # lr_end = 0.1
    # lr_steps = 100
