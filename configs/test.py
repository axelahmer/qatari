from .machado import MachadoConfig


class TestConfig(MachadoConfig):
    """
    configuration used for testing purposes.
    """
    def __init__(self):
        super().__init__()
        self.learning_start = 5_000
        self.debug = True  # whether to print state and observation images to tensorboard
        self.display = True  # display game and max q plot
        # self.device = 'cuda:0'  # what device to do learning updates with
