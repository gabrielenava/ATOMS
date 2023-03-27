from atoms.atoms_helpers import Helpers


class OneClassSVM:
    """
    OneClassSVM class:
    """

    def __init__(self, debug=False):

        if debug:
            self.logger = Helpers.init_logger()

    def __str__(self):
        return f"OneClassSVM class object"

    def setup(self):
        return
