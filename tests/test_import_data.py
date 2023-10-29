# Testing of the ImportData class from the ATOMS package
import os
import unittest
from atoms.import_data import ImportData
from os.path import join, dirname, abspath


class TestImportData(unittest.TestCase):

    def test_import_data(self):

        i = ImportData(debug=True)

        # get data path and data name, and list all variables to be imported

        # warning: if a variable is a string, it will be addressed under the keyword 'none'. For the moment, the class
        # cannot handle it. Just avoid to use strings as variables
        current_folder_path = dirname(abspath(__file__))
        data_path_and_name = join(current_folder_path, 'test_data/dataset_test_bench_P100-4102.mat')

        variables_list = ['t_step', 'P_atm', 'T_atm', 'time', 'turbine_status', 'egt_temperature', 'throttle',
                          'fuel_consumed', 'rpm_measured', 'rpm_desired', 'thrust']

        # load data in the class object
        i.load(data_path_and_name, 't_step')
        i.load(data_path_and_name, variables_list)

        # plot the loaded data. Single and multiple plot
        plt = i.plot(['time', 'fuel_consumed'])
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

        plt = i.plot([['time', 'turbine_status'], ['time', 'egt_temperature'], ['time', 'throttle'],
                      ['time', 'fuel_consumed'], ['time', 'rpm_measured'], ['time', 'rpm_desired'], ['time', 'thrust']])
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

        # test subtract initial value and normalize
        i.normalize(['egt_temperature', 'fuel_consumed'])
        i.normalize('rpm_measured')
        plt = i.plot(['time', 'rpm_measured'])
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

        # test split dataset
        i.split(['time', 'egt_temperature', 'fuel_consumed'], [0.5, 0.5, 0.33], 'rpm_measured')
        i.split(['time', 'rpm_desired', 'rpm_measured'], 10, 'time')
        i.split('thrust', 10.33, 'time')
        i.split('thrust', [10.01, 34], 'time')

        # test plot datasets
        plt = i.plot(['time', 'egt_temperature'], 'dataset_2')
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

        plt = i.plot([['time', 'rpm_desired'], ['time', 'rpm_measured']], 'dataset_4')
        plt.show(block=False)
        plt.pause(1)
        plt.close('all')

        # test save data
        i.save(['rpm_desired', 'rpm_measured'])
        i.save('rpm_desired', 'dataset_4')

        if os.path.exists('rpm_desired.npy'):
            os.remove('rpm_desired.npy')
        if os.path.exists('rpm_measured.npy'):
            os.remove('rpm_measured.npy')

        # verify if the object of the class is correct
        self.assertEqual(i.data['t_step'], [0.01])
        self.assertEqual(i.variables_list, variables_list)
        self.assertEqual(list(i.datasets.keys()), ['dataset_0', 'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4',
                                                   'dataset_5', 'dataset_6', 'dataset_7', 'dataset_8', 'dataset_9',
                                                   'dataset_10'])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestImportData('test_import_data'))
