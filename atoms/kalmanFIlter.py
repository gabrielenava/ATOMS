import numpy


class KalmanFilter:
    """
    KalmanFilter class:
    """

    def __init__(self):
        self.variables = {}

    def __str__(self):
        return

    def setup(self, variables):
        """
        Load all process and filter quantities and store them in the class variables.
        """
        self.variables.update({'X': variables['X']})
        self.variables.update({'A': variables['A']})
        self.variables.update({'B': variables['B']})
        self.variables.update({'U': variables['U']})
        self.variables.update({'P': numpy.zeros(variables['Q'].size)})
        self.variables.update({'Q': variables['Q']})

    def predict(self):
        """
        Implements the prediction phase of the KF algorithm. Legend:
        - X = mean state estimate at (k-1)
        - P = state covariance at (k-1)
        - A = transition matrix
        - B = input matrix
        - U = control input
        - Q = process noise covariance matrix
        """

        # calculate X(k) from (k-1) quantities
        self.variables['X'] = numpy.dot(self.variables['A'], self.variables['X']) + numpy.dot(self.variables['B'],
                                                                                              self.variables['U'])
        # calculate the state covariance P(k) from (k-1) quantities
        self.variables['P'] = numpy.dot(self.variables['A'], numpy.dot(self.variables['P'],
                                                                       self.variables['A'].T)) + self.variables['Q']
