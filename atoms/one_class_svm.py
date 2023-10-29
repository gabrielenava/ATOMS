from atoms import atoms_helpers
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class OneClassSupportVectorMachine:
    """
    OneClassSupportVectorMachine: wrapper of One-Class Support Vector Machine from scikit_learn. See also
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html for more details.
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.svm = None
        self.helpers = atoms_helpers.Helpers()

        if debug:
            self.logger = atoms_helpers.Helpers.init_logger()

    def __str__(self):
        return f" OneClassSupportVectorMachine class object. \n"

    def init(self, nu=0.5, kernel='rbf', gamma='scale'):
        """
        init: initialize the one-class SVM classifier.
        :param nu: (default: 0.5)
        :param kernel: (default: 'rbf')
        :param gamma: (default: 'scale')
        See also the documentation of OneClassSVM class from scikit_learn for details.
        """
        self.svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

        if self.debug:
            self.logger.debug('[init]: SVM classifier initialized.')

    def train(self, x_train):
        """
        train: trains the one-class SVM classifier.
        :param x_train: the training dataset
        """
        self.svm.fit(x_train)

        if self.debug:
            self.logger.debug('[train]: SVM classifier trained.')

    def predict(self, x_test):
        """
        predict: tests the one-class SVM classifier.
        :param x_test: the test dataset.
        :return: prediction_labels (the predicted labels for the test dataset).
        """
        prediction_labels = self.svm.predict(x_test)

        if self.debug:
            self.logger.debug('[predict]: test of SVM classifier completed.')

        return prediction_labels

    def evaluate(self, true_labels, predicted_labels, metric):
        """
        evaluate: tests the one-class SVM classifier.
        :param true_labels: the real labels of the dataset.
        :param predicted_labels: the predicted labels of the dataset.
        :param metric: the selected metric for evaluating the algorithm performance. Available metrics: precision,
        recall, f1, auc_roc.
        :return metric_result: the obtained score for the selected metric.
        """
        available_metrics = ['precision', 'recall', 'f1', 'auc_roc']
        is_str = self.helpers.check_if_list_or_string(metric)
        metric_result = []

        if is_str == 'str':

            is_metric = self.helpers.check_if_data_in_list(available_metrics, metric)
            if is_metric:
                # compute the score for the user-specified metric
                if metric == 'precision':
                    metric_result = precision_score(true_labels, predicted_labels)
                elif metric == 'recall':
                    metric_result = recall_score(true_labels, predicted_labels)
                elif metric == 'f1':
                    metric_result = f1_score(true_labels, predicted_labels)
                elif metric == 'auc_roc':
                    metric_result = roc_auc_score(true_labels, predicted_labels)
            else:
                raise ValueError('[evaluate]: the user provided metric is not among available metrics.')
        else:
            raise ValueError('[evaluate]: the user provided metric is not a string.')

        if self.debug:
            self.logger.debug(f"[evaluate]: metric {metric} evaluated.")

        return metric_result
