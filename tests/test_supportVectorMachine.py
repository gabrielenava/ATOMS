# Testing of the LinearMPC class from the ATOMS package
import unittest
from atoms.supportVectorMachine import OneClassSVM


class TestOneClassSVM(unittest.TestCase):

    def test_OneClassSVM(self):

        svm = OneClassSVM()

        # verify if the object of the class is correct
        self.assertEqual('', '')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestOneClassSVM('test_OneClassSVM'))
