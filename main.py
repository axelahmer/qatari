import sys
from core.dqn_learner import DQNLearner
from configs import config_dict

if __name__ == '__main__':
    game = None
    config = None

    # read arguments
    for v in sys.argv[1:]:
        arg_name, arg_val = v.split("=")
        if arg_name == 'game':
            game = arg_val
        elif arg_name == 'config':
            config = config_dict[arg_val]()
        else:
            print('unrecognized argument. expected: "python main.py game=____ config=_____"')

    model = DQNLearner(game, config)
    model.run()
