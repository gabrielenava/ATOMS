import logging


class Helpers:
    """
    Helpers class: collection of methods to be used in the other classes of the atoms package.
    """
    def __init__(self):
        self.helpers_msg = 'ATOMS helpers object.'

    def __str__(self):
        return f" {self.helpers_msg}"

    @staticmethod
    def init_logger():

        # create a logger object
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create a console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # add console handler to the logger
        logger.addHandler(ch)

        return logger
