# Testing of the OneClassSupportVectorMachine class from ATOMS package
import unittest
import numpy as np
from atoms.one_class_svm import OneClassSupportVectorMachine


class TestSVM(unittest.TestCase):

    def test_svm(self):

        x = [[0], [0.44], [0.45], [0.46], [1]]

        svm = OneClassSupportVectorMachine(debug=True)
        svm.init(nu=0.05, kernel='rbf', gamma='auto')
        svm.train(x)

        predicted_labels = svm.predict(x)

        # test metrics for SVM prediction
        true_labels = np.array([-1, 1, 1, 1, -1])
        precision = svm.evaluate(true_labels, predicted_labels, 'precision')
        recall = svm.evaluate(true_labels, predicted_labels, 'recall')
        f1_score = svm.evaluate(true_labels, predicted_labels, 'f1')
        auc_roc = svm.evaluate(true_labels, predicted_labels, 'auc_roc')

        # verify if the object of the class is correct
        self.assertEqual(predicted_labels[0], -1)
        self.assertEqual(predicted_labels[1],  1)
        self.assertEqual(predicted_labels[2],  1)
        self.assertEqual(predicted_labels[3],  1)
        self.assertEqual(predicted_labels[4], -1)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(f1_score, 1.0)
        self.assertEqual(auc_roc, 1.0)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestSVM('test_svm'))
