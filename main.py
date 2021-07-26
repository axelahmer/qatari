import argparse
import sys
from core.dqn_learner import DQNLearner
from configs import config_dict
from modules import module_dict

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Process Qatari training arguments.')
    parser.add_argument('--config', type=str, default='machado', choices=config_dict.keys())
    parser.add_argument('--env_type', type=str, choices=['atari', 'procgen'])
    parser.add_argument('--game', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--qnet', type=str, choices=module_dict.keys())
    parser.add_argument('--optim', type=str, choices=['adam', 'rms_prop'])
    parser.add_argument('--double', type=bool, default=False)
    args = parser.parse_args()

    # make config
    config = config_dict[args.config]()

    # overwrite config with args if specified
    if args.env_type is not None:
        config.env_type = args.env_type
    if args.game is not None:
        config.game = args.game
    if args.seed is not None:
        config.seed = args.seed
    if args.qnet is not None:
        config.qnet = args.qnet
    if args.optim is not None:
        config.optimizer = args.optim
    if args.double is not None:
        config.double_dqn = args.double

    # load and run model
    model = DQNLearner(config)
    model.run()
