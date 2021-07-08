from .nature import NatureNet
from .summer import SummerNet
from .qmixer import MixerNet
from .split1 import SplitNet1

module_dict = {}

# not dumb
module_dict["nature"] = NatureNet

# kinda dumb
module_dict["mixer"] = MixerNet

# very dumb
module_dict["summer"] = SummerNet

module_dict["split1"] = SplitNet1