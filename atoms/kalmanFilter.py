import numpy
from numpy import dot, sum, tile
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
                raise ValueError('Required variable', var, 'not found in the input dictionary.')

        # add matrix P to the variables dictionary
        self.variables.update({'P': numpy.zeros(variables['Q'].shape)})

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
        self.variables['X'] = dot(self.variables['A'], self.variables['X']) + dot(self.variables['B'],
                                                                                  self.variables['U'])
        # calculate the state covariance P(k) from (k-1) quantities
        self.variables['P'] = dot(self.variables['A'], dot(self.variables['P'],
                                                           self.variables['A'].T)) + self.variables['Q']

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
        y_mean_predicted = dot(self.variables['C'], self.variables['X'])
        y_covariance = self.variables['R'] + dot(self.variables['C'], dot(self.variables['P'], self.variables['C'].T))
        k_gain = dot(self.variables['P'], dot(self.variables['C'].T, inv(y_covariance)))

        # correct the predicted state and covariance matrix
        self.variables['X'] = self.variables['X'] + dot(k_gain, (y_measured - y_mean_predicted))
        self.variables['P'] = self.variables['P'] - dot(k_gain, dot(y_covariance, k_gain.T))
        x_estimated = self.variables['X']

        # calculate the predictive probability of the measurements
        y_predictive_prob = self.__gauss_pdf(y_measured, y_mean_predicted, y_covariance)

        return x_estimated, y_predictive_prob

    @staticmethod
    def __gauss_pdf(y_measured, y_mean_predicted, y_covariance):

        # see also https://arxiv.org/pdf/1204.0375.pdf
        if y_mean_predicted.shape[1] == 1:
            delta_y = y_measured - tile(y_mean_predicted, y_measured.shape[1])
            E = 0.5 * sum(delta_y * (dot(inv(y_covariance), delta_y)), axis=0)
            E = E + 0.5 * y_mean_predicted.shape[0] * numpy.log(2 * numpy.pi) + 0.5 * numpy.log(det(y_covariance))
            y_predictive_prob = numpy.exp(-E)

        elif y_measured.shape[1] == 1:
            delta_y = tile(y_measured, y_mean_predicted.shape[1]) - y_mean_predicted
            E = 0.5 * sum(delta_y * (dot(inv(y_covariance), delta_y)), axis=0)
            E = E + 0.5 * y_mean_predicted.shape[0] * numpy.log(2 * numpy.pi) + 0.5 * numpy.log(det(y_covariance))
            y_predictive_prob = numpy.exp(-E)

        else:
            delta_y = y_measured - y_mean_predicted
            E = 0.5 * dot(delta_y.T, dot(inv(y_covariance), delta_y))
            E = E + 0.5 * y_mean_predicted.shape[0] * numpy.log(2 * numpy.pi) + 0.5 * numpy.log(det(y_covariance))
            y_predictive_prob = numpy.exp(-E)

        return y_predictive_prob
