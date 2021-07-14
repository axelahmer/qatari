from .nature import NatureNet
from .summer import SummerNet
from .qmixer import MixerNet
from .split1 import SplitNet1
from .split2 import SplitNet2

module_dict = {"nature": NatureNet,
               "mixer": MixerNet,
               "summer": SummerNet,
               "split1": SplitNet1,
               "split2": SplitNet2}

