import numpy


class KalmanFilter:
    """
    KalmanFilter class: implementation of Kalman filter for discrete, time invariant linear systems. Estimates the state
    X(k) of the following system:

      X(k) = A*X(k-1) + B*U(k) + W(k-1)
      Y(k) = C*X(k) + V(k)

    with V, W white, uncorrelated, zero mean noise on the process and on the measurements. The user is required to tune
    the covariance matrices Q and R for the process and measurement noise, respectively.
    """

    def __init__(self):
        self.variables = {}

    def __str__(self):
        return f" KalmanFilter class object \n" \
               f" Stored variables: {self.variables}"

    def setup(self, variables):
        """
        Load the process, measurements and covariances. See the class description to know which variables are needed.
        """
        expected_variables = ['X', 'A', 'B', 'U', 'Q']
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
        Implement the prediction phase of the KF algorithm. Legend:
        - X = mean state estimate at (k-1)
        - P = state covariance at (k-1)
        - A = transition matrix
        - B = input matrix
        - U = control input at (k)
        - Q = process noise covariance matrix
        """

        # calculate X(k) from (k-1) quantities
        self.variables['X'] = numpy.dot(self.variables['A'], self.variables['X']) + numpy.dot(self.variables['B'],
                                                                                              self.variables['U'])
        # calculate the state covariance P(k) from (k-1) quantities
        self.variables['P'] = numpy.dot(self.variables['A'], numpy.dot(self.variables['P'],
                                                                       self.variables['A'].T)) + self.variables['Q']

    def update(self):
        """
        Implement the update phase of the KF algorithm. Legend:
        """
        return
