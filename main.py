import sys
from core.dqn_learner import DQNLearner
from configs import config_dict

if __name__ == '__main__':
    game = None
    qnet = None
    config = None

    # read arguments
    for v in sys.argv[1:]:
        arg_name, arg_val = v.split("=")
        if arg_name == 'game':
            game = arg_val
        elif arg_name == 'qnet':
            qnet = arg_val
        elif arg_name == 'config':
            config = config_dict[arg_val]()
        else:
            print('unrecognized argument.\n\nexpected: "python main.py game=GAME_NAME module=MODULE_NAME config=CONFIG_NAME"')

    # overwrite config attributes
    config.game = game
    config.qnet = qnet

    # load and run model
    model = DQNLearner(config)
    model.run()
