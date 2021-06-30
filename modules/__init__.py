from .nature import NatureNet
from .summer import SummerNet
from .qmixer import MixerNet


module_dict = {}

# not dumb
module_dict["nature"] = NatureNet

# kinda dumb
module_dict["mixer"] = MixerNet

# very dumb
module_dict["summer"] = SummerNet
