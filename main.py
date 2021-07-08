import sys
from core.dqn_learner import DQNLearner
from configs import config_dict

if __name__ == '__main__':
    game = None
    qnet = None
    optim = None
    seed = None
    config = None

    # read arguments
    for v in sys.argv[1:]:
        arg_name, arg_val = v.split("=")
        if arg_name == 'game':
            game = arg_val
        elif arg_name == 'qnet':
            qnet = arg_val
        elif arg_name == 'optim':
            optim = arg_val
        elif arg_name == 'seed':
            seed = int(arg_val)
        elif arg_name == 'config':
            config = config_dict[arg_val]()
        else:
            print('error.\n\nexpected use: "python main.py game=GAME_NAME qnet=MODULE_NAME optim=OPTIMIZER_NAME seed=SEED config=CONFIG_NAME"')

    if config is None:
        config = config_dict['machado']()

    # overwrite config attributes
    if game is not None:
        config.game = game

    if qnet is not None:
        config.qnet = qnet

    if optim is not None:
        config.optimizer = optim

    if seed is not None:
        config.seed = seed

    # load and run model
    model = DQNLearner(config)
    model.run()
