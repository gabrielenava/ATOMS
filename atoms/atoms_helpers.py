import logging


class Helpers:
    """
    Helpers class: collection of methods to be used in the other classes of the ATOMS package.
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

    @staticmethod
    def check_if_list_or_string(data):

        if isinstance(data, str):
            return 'str'
        elif isinstance(data, list):
            return 'list'
        else:
            raise ValueError('[check_if_list_or_string] input data is not a list or string.')

    @staticmethod
    def check_if_data_in_list(list_name, data_name):
        if data_name in list_name:
            return True
        else:
            return False
