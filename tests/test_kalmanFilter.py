# Testing of the KalmanFilter class from the ATOMS package
import unittest
import numpy as np
from atoms.kalmanFilter import KalmanFilter


class TestKalmanFilter(unittest.TestCase):

    def test_KalmanFilter(self):

        # initialization of state matrices
        A = np.eye(2)
        X = np.array([[0.1], [0.1]])
        B = np.ones((X.shape[0], 1))
        U = 0.5

        # process noise covariance matrix and initial state covariance
        Q = np.eye(X.shape[0])
        P = np.diag((0.01, 0.01))

        # measurement matrices (state X plus a random gaussian noise)
        Y = np.array([X[0, 0] + 0.001, X[1, 0] + 0.001])
        Y = Y.reshape(-1, 1)
        C = np.eye(2)

        # measurement noise covariance
        R = np.eye(Y.shape[0])

        var = {}
        var.update({'X': X})
        var.update({'A': A})
        var.update({'B': B})
        var.update({'U': U})
        var.update({'Q': Q})
        var.update({'C': C})
        var.update({'R': R})

        kf = KalmanFilter()
        kf.setup(var)
        kf.predict()
        x_est, y_predict = kf.update(Y)

        # verify if the object of the class is correct
        self.assertEqual(x_est[0], 0.3505)
        self.assertEqual(x_est[1], 0.3505)
        self.assertEqual(y_predict[0], 0.07026195923972386)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestKalmanFilter('test_KalmanFilter'))
