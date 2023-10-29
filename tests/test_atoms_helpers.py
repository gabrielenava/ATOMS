# Testing of the Helpers class from the ATOMS package
import unittest
from atoms.atoms_helpers import Helpers


class TestHelpers(unittest.TestCase):

    def test_helpers(self):

        h = Helpers()

        type_str = h.check_if_list_or_string('a')
        type_list = h.check_if_list_or_string(['a', 'b', 'c'])

        is_in_list = h.check_if_data_in_list(['a', 'b', 'c'], 'a')
        not_in_list = h.check_if_data_in_list(['a', 'b', 'c'], 'd')

        logger = h.init_logger()
        logger.debug('tested logger functionality.')

        # verify if the object of the class is correct
        self.assertEqual(type_str, 'str')
        self.assertEqual(type_list, 'list')
        self.assertEqual(is_in_list, True)
        self.assertEqual(not_in_list, False)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestHelpers('test_helpers'))
