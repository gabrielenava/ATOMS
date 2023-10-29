import traceback
import numpy as np
import matplotlib.pyplot as plt
from atoms import import_data
from atoms import atoms_helpers
from atoms import one_class_svm
from os.path import join, dirname, abspath

logger = atoms_helpers.Helpers.init_logger()


def load_data(data_file_name, debug=False):
    """
    Load data from .mat file and return the rpm vectors.
    """
    if debug:
        logger.info(f'Loading data from {data_file_name}')

    data_obj = import_data.ImportData()
    current_folder_path = dirname(abspath(__file__))
    data_path_and_name = join(current_folder_path, f'data/{data_file_name}')

    variables_list = ['t_step', 'P_atm', 'T_atm', 'time', 'turbine_status', 'egt_temperature', 'throttle',
                      'fuel_consumed', 'rpm_measured', 'rpm_desired', 'rpm_simulated', 'thrust', 'pump_voltage']

    # load data in the class object and select part of RPM vector for the training
    data_obj.load(data_path_and_name, variables_list)
    data_obj.split(['rpm_measured', 'rpm_simulated'], 95, 'time')
    split_data = data_obj.datasets['dataset_1']
    measured_rpm = split_data['rpm_measured']
    simulated_rpm = split_data['rpm_simulated']

    if debug:
        logger.info(f'Data loaded from {data_file_name}')

    return measured_rpm, simulated_rpm


def run_svm_algorithm(measured_rpm, simulated_rpm, debug=False):
    """
    Implement one-class SVM to detect anomalies in the measured RPM.
    """
    if debug:
        logger.info(f'Running SVM algorithm...')

    # settings for the one-class SVM algorithm
    nu = 0.00001
    kernel = 'rbf'
    gamma = 'scale'
    evaluation_metric = 'f1'

    # initialize the problem
    svm = one_class_svm.OneClassSupportVectorMachine()
    svm.init(nu, kernel, gamma)

    # combine the data into a single feature matrix and train the algorithm
    x_train = np.column_stack((measured_rpm, simulated_rpm))
    svm.train(x_train)

    # add artificial anomalies to the measured RPM data, and define the corresponding labels
    measured_rpm[10000:15000] = measured_rpm[10000:15000] * 0.7
    y_true = np.ones(len(measured_rpm))
    y_true[10000:15000] = y_true[10000:15000] - 2

    # predict the anomalies in the measured RPM data
    x_test = np.column_stack((measured_rpm, simulated_rpm))
    y_predicted = svm.predict(x_test)

    # evaluate the results according to the user-specified metrics
    score_metric = svm.evaluate(y_true, y_predicted, evaluation_metric)

    return y_true, y_predicted, score_metric


def run_anomaly_detection(debug=False):
    """
    Run the anomaly detection algorithm and print results.
    """
    if debug:
        logger.info('Anomaly detection for jet engines with ATOMS library.')

    measured_rpm, simulated_rpm, = load_data('dataset_test_bench_P220-688_exp_18-58.mat', debug)
    y_true, y_predicted, score_metric = run_svm_algorithm(measured_rpm, simulated_rpm, debug)

    # Plot the results
    plt.figure()
    plt.plot(measured_rpm, label='Measured RPM')
    plt.plot(simulated_rpm, label='Simulated RPM')
    plt.plot(np.where(y_true == -1)[0], measured_rpm[y_true == -1], 'rx', label='True Anomalies')
    plt.plot(np.where(y_predicted == -1)[0], measured_rpm[y_predicted == -1], 'g.', label='Predicted Anomalies')
    plt.legend()
    plt.show()

    # Print the metrics results
    print(f"Prediction score: {score_metric}")


if __name__ == "__main__":
    try:
        run_anomaly_detection(debug=True)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise
