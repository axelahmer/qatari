import argparse
import sys
from core.dqn_learner import DQNLearner
from configs import config_dict
from modules import module_dict

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Process Qatari training arguments.')
    parser.add_argument('--config', type=str, default='machado', choices=config_dict.keys())
    parser.add_argument('--game', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--qnet', type=str, choices=module_dict.keys())
    parser.add_argument('--optim', type=str, choices=['adam', 'rms_prop'])
    parser.add_argument('--double', type=bool, default=False)
    args = parser.parse_args()

    # make config
    config = config_dict[args.config]()

    # overwrite config with args if specified
    if hasattr(args, 'game'):
        config.game = args.game
    if hasattr(args, 'seed'):
        config.seed = args.seed
    if hasattr(args, 'qnet'):
        config.qnet = args.qnet
    if hasattr(args, 'optim'):
        config.optimizer = args.optim
    if hasattr(args, 'double'):
        config.double_dqn = args.double

    # load and run model
    model = DQNLearner(config)
    model.run()
