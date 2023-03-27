# Testing of the LinearMPC class from the ATOMS package
import unittest
from atoms.linearMPC import LinearMPC


class TestLinearMPC(unittest.TestCase):

    def test_LinearMPC(self):

        opti = LinearMPC()

        # verify if the object of the class is correct
        self.assertEqual('', '')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestLinearMPC('test_LinearMPC'))
