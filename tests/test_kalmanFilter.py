# Testing of the KalmanFilter class from the ATOMS package
import unittest
import numpy as np
from atoms.kalmanFilter import KalmanFilter


class TestKalmanFilter(unittest.TestCase):

    def test_KalmanFilter(self):
        print('Run KF class test.')

        # initialization of state matrices
        A = np.eye(2)
        X = np.array([[0.1], [0.1]])
        B = np.ones((2, 1))  # 2 rows (same as X), 1 column
        U = np.array([[0.5]])  # single control input as a column vector

        # process noise covariance matrix and initial state covariance
        Q = np.eye(2)
        P = np.diag((0.01, 0.01))

        # measurement matrices (state X plus a random gaussian noise)
        Y = np.array([[X[0, 0] + 0.001], [X[1, 0] + 0.001]])  # 2x1 matrix
        C = np.eye(2)

        # measurement noise covariance
        R = np.eye(2)

        var = {
            'X': X,
            'A': A,
            'B': B,
            'U': U,
            'Q': Q,
            'C': C,
            'R': R
        }

        kf = KalmanFilter()
        kf.setup(var)
        kf.predict()
        x_est, y_predict = kf.update(Y)

        # verify if the object of the class is correct
        np.testing.assert_almost_equal(x_est[0], 0.1014, decimal=4)
        np.testing.assert_almost_equal(x_est[1], 0.1014, decimal=4)
        np.testing.assert_almost_equal(y_predict[0], 0.0001588, decimal=7)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestKalmanFilter('test_KalmanFilter'))
    unittest.TextTestRunner().run(suite)
