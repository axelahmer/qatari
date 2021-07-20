from .nature import NatureNet
from .summer import SummerNet
from .qmixer import MixerNet
from .split1 import SplitNet1
from .split2 import SplitNet2
from .dueling_experts import DuelingAdvantages, DuelingTFAS
from .boosters import AdvantageBooster1
from dueling_vanilla import DuelingDQN

module_dict = {"nature": NatureNet,
               "mixer": MixerNet,
               "summer": SummerNet,
               "split1": SplitNet1,
               "split2": SplitNet2,
               "duelvanilla": DuelingDQN,
               "duela": DuelingAdvantages,
               "dueltfas": DuelingTFAS,
               "boost1": AdvantageBooster1}
