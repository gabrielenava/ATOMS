# Testing of the Helpers class from the ATOMS package
import unittest
from atoms.atoms_helpers import Helpers


class TestHelpers(unittest.TestCase):

    def test_helpers(self):

        h = Helpers()
        logger = h.init_logger()
        logger.debug('tested logger functionality.')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestHelpers('test_helpers'))
