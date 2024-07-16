import numpy as np
from numpy.linalg import inv, det
from atoms.atoms_helpers import Helpers


class KalmanFilter:
    """
    KalmanFilter class: implementation of Kalman filter for discrete, time invariant linear systems. Estimates the state
    X(k) of the following system:

      X(k) = A*X(k-1) + B*U(k) + W(k-1)
      Y(k) = C*X(k) + V(k)

    with V, W, white, uncorrelated, zero mean noise on the process and on the measurements. The user is required to tune
    the covariance matrices Q and R for the process and measurements noise, respectively.
    """

    def __init__(self, debug=False):
        self.variables = {}
        self.debug = debug

        if debug:
            self.logger = Helpers.init_logger()

    def __str__(self):
        return f" KalmanFilter class object \n" \
               f" Stored variables: {self.variables.keys()}"

    def setup(self, variables):
        """
        Load the process variables, measurements, and covariance matrices. See the class description to know exactly
        which variables are needed. variables is a dictionary with the expected variables as keys.
        """
        expected_variables = ['X', 'A', 'B', 'U', 'C', 'Q', 'R']
        var_keys = list(variables.keys())

        for var in expected_variables:
            if var in var_keys:
                self.variables.update({var: variables[var]})
            else:
                raise ValueError(f'Required variable {var} not found in the input dictionary.')

        # Initialize matrix P to the identity matrix scaled by a large number
        self.variables.update({'P': np.eye(variables['Q'].shape[0]) * 1000})

    def predict(self):
        """
        Implement the prediction phase of the KF algorithm. Variables needed:
        - X = mean state estimate at (k-1)
        - A = transition matrix
        - B = input matrix
        - U = control input at (k)
        - P = state covariance at (k-1)
        - Q = process noise covariance matrix
        """
        # calculate X(k) from (k-1) quantities
        self.variables['X'] = self.variables['A'] @ self.variables['X'] + self.variables['B'] @ self.variables['U']

        # calculate the state covariance P(k) from (k-1) quantities
        self.variables['P'] = self.variables['A'] @ self.variables['P'] @ self.variables['A'].T + self.variables['Q']

        if self.debug:
            self.logger.info(f"Predicted state X: {self.variables['X']}")
            self.logger.info(f"Predicted state covariance P: {self.variables['P']}")

    def update(self, y_measured):
        """
        Implement the update phase of the KF algorithm. The input y_measured is Y(k) (measurement at time k).
        Variables needed:
        - X = predicted state estimate at (k)
        - P = predicted state covariance at (k)
        - C = state to measurements matrix
        - R = measurements noise covariance matrix
        Returns the predictive probability (likelihood) of the measurements.
        """
        y_mean_predicted = self.variables['C'] @ self.variables['X']
        y_covariance = self.variables['R'] + self.variables['C'] @ self.variables['P'] @ self.variables['C'].T
        k_gain = self.variables['P'] @ self.variables['C'].T @ inv(y_covariance)

        # correct the predicted state and covariance matrix
        self.variables['X'] = self.variables['X'] + k_gain @ (y_measured - y_mean_predicted)
        self.variables['P'] = self.variables['P'] - k_gain @ self.variables['C'] @ self.variables['P']
        x_estimated = self.variables['X']

        # calculate the predictive probability of the measurements
        y_predictive_prob = self.__gauss_pdf(y_measured, y_mean_predicted, y_covariance)

        if self.debug:
            self.logger.info(f"Updated state X: {self.variables['X']}")
            self.logger.info(f"Updated state covariance P: {self.variables['P']}")
            self.logger.info(f"Kalman Gain K: {k_gain}")
            self.logger.info(f"Predictive probability: {y_predictive_prob}")

        return x_estimated, y_predictive_prob

    @staticmethod
    def __gauss_pdf(y_measured, y_mean_predicted, y_covariance):
        delta_y = y_measured - y_mean_predicted
        exponent = -0.5 * delta_y.T @ inv(y_covariance) @ delta_y
        normalization = 0.5 * len(y_mean_predicted) * np.log(2 * np.pi) + 0.5 * np.log(det(y_covariance))
        y_predictive_prob = np.exp(exponent - normalization)
        return y_predictive_prob
