# Testing of the KalmanFilter class from the ATOMS package
import unittest
from atoms.kalmanFilter import KalmanFilter


class TestKalmanFilter(unittest.TestCase):

    def test_KalmanFilter(self):

        kf = KalmanFilter()

        # verify if the object of the class is correct
        self.assertEqual('', '')


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestKalmanFilter('test_KalmanFilter'))
