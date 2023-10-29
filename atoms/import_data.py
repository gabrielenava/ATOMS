import numpy
from scipy import io
from atoms import atoms_helpers
from matplotlib import pyplot as plt


class ImportData:
    """
    ImportData class: import, modify and plot data from MATLAB .mat files.
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.data = {}
        self.datasets = {}
        self.counter = 0
        self.variables_list = []
        self.helpers = atoms_helpers.Helpers()

        if debug:
            self.logger = atoms_helpers.Helpers.init_logger()

    def __str__(self):
        return f" ImportData class object \n" \
               f" Loaded data: {self.variables_list} \n" \
               f" Generated datasets: {self.datasets.keys()}"

    def load(self, data_path_and_name, variables_list):
        """
        Load data from .mat file.
        :param data_path_and_name: a string with the path of the folder where data are stored, joined with the name of
        the data to load.
        :param variables_list: the list of variables contained in the loaded file that the user would like to import.
        """
        self.helpers.check_if_list_or_string(data_path_and_name)
        var_type = self.helpers.check_if_list_or_string(variables_list)
        mat_data = io.loadmat(data_path_and_name)

        # generate the dict containing all data in the file
        if len(variables_list) == 0:
            raise ValueError('[load]: variables_list cannot be empty!')
        else:
            if var_type == 'str':
                self.data.update({variables_list: mat_data[variables_list]})
                if self.debug:
                    self.logger.debug(f'[load]: data {variables_list} added to self.data.')
            elif var_type == 'list':
                for var_name in variables_list:
                    self.data.update({var_name: mat_data[var_name]})
                    if self.debug:
                        self.logger.debug(f'[load]: data {var_name} added to self.data.')

        self.variables_list = variables_list

    def plot(self, x_y_axis_pairs, dataset_name=''):
        """
        Plot the loaded data.
        :param x_y_axis_pairs: a nested list containing the pair of x and y axes for each plot.
        :param dataset_name: the name of the dataset from which to plot the data. If empty, the variable 'self.data' is
        used by default.
        """
        self.helpers.check_if_list_or_string(x_y_axis_pairs)
        n_subplots = len(x_y_axis_pairs)
        i = 1

        if len(dataset_name) == 0:
            # use 'self.data' to search for and plot the x_y_axis_pairs
            data_to_plot = self.data
        else:
            dataset_name_type = self.helpers.check_if_list_or_string(dataset_name)
            if dataset_name_type == 'str':
                is_dataset = self.helpers.check_if_data_in_list(self.datasets.keys(), dataset_name)
                if is_dataset:
                    data_to_plot = self.datasets[dataset_name]
                else:
                    raise ValueError(f'[plot]: dataset name {dataset_name} not found in self.datasets keys.')
            else:
                raise ValueError('[plot]: dataset_name is not a valid string.')

        if isinstance(x_y_axis_pairs[0], list):
            for x_y_pair in x_y_axis_pairs:
                if x_y_pair[0] in self.variables_list and x_y_pair[1] in self.variables_list:
                    # check if we need to create a new figure (no more than 5 plots allowed per figure)
                    if i > 5:
                        plt.figure()
                        i = 1
                    plt.subplot(min(n_subplots, 5), 1, i)
                    plt.plot(data_to_plot[x_y_pair[0]], data_to_plot[x_y_pair[1]])
                    plt.xlabel(x_y_pair[0])
                    plt.ylabel(x_y_pair[1])
                    plt.grid(True)
                    i += 1
                else:
                    raise ValueError('[plot]: x_y_axis_pairs contains labels that are not in the variables list.')
        elif len(x_y_axis_pairs) == 2:
            # the user may have provided a single pair of variables as a list (not nested). This is also ok
            if x_y_axis_pairs[0] in self.variables_list and x_y_axis_pairs[1] in self.variables_list:
                plt.subplot(1, 1, i)
                plt.plot(data_to_plot[x_y_axis_pairs[0]], data_to_plot[x_y_axis_pairs[1]])
                plt.xlabel(x_y_axis_pairs[0])
                plt.ylabel(x_y_axis_pairs[1])
                plt.grid(True)
            else:
                raise ValueError('[plot]: x_y_axis_pairs contains labels that are not in the variables list.')
        else:
            raise ValueError('[plot]: x_y_axis_pairs is not a valid list.')

        return plt

    def normalize(self, data_list):
        """
        Normalize the data specified in data_list. Normalization is done by dividing all data for the max abs value.
        :param data_list: the list of data to normalize.
        """
        data_type = self.helpers.check_if_list_or_string(data_list)

        if data_type == 'str':
            if self.helpers.check_if_data_in_list(self.variables_list, data_list):
                max_value = abs(max(self.data[data_list], key=abs))
                if max_value > 0:
                    self.data[data_list] = self.data[data_list] / max_value
                    if self.debug:
                        self.logger.debug(f'[normalize]: data {data_list} normalized.')
                else:
                    raise ValueError('[normalize]: max abs value of selected data is 0.')
            else:
                raise ValueError('[normalize]: data not found in the variable list.')
        elif data_type == 'list':
            for data_name in data_list:
                if self.helpers.check_if_data_in_list(self.variables_list, data_name):
                    max_value = abs(max(self.data[data_name], key=abs))
                    if max_value > 0:
                        self.data[data_name] = self.data[data_name] / max_value
                        if self.debug:
                            self.logger.debug(f'[normalize]: data {data_name} normalized.')
                    else:
                        raise ValueError('[normalize]: max abs value of selected data is 0.')
                else:
                    raise ValueError('[normalize]: data not found in the variable list.')

    def save(self, data_list, dataset_name=''):
        """
        Save the selected data.
        :param data_list: a list with the name of data to be saved.
        :param dataset_name: the name of the dataset from which to save the data. If empty, 'self.data' is selected.
        """
        self.helpers.check_if_list_or_string(data_list)

        if len(dataset_name) == 0:
            # use 'self.data' to search for data to save
            data_to_save = self.data
        else:
            dataset_name_type = self.helpers.check_if_list_or_string(dataset_name)
            if dataset_name_type == 'str':
                is_dataset = self.helpers.check_if_data_in_list(self.datasets.keys(), dataset_name)
                if is_dataset:
                    data_to_save = self.datasets[dataset_name]
                else:
                    raise ValueError(f'[save]: dataset name {dataset_name} not found in self.datasets keys.')
            else:
                raise ValueError('[save]: dataset_name is not a valid string.')

        if isinstance(data_list, list):
            for data_name in data_list:
                if data_name in data_to_save.keys():
                    numpy.save(f'{data_name}.npy', data_to_save[data_name])
                else:
                    raise ValueError('[save]: data_list contains labels that are not in the variables list.')

        elif isinstance(data_list, str):

            if data_list in data_to_save.keys():
                numpy.save(f'{data_list}.npy', data_to_save[data_list])
            else:
                raise ValueError('[save]: data_name not found in the variables list.')
        else:
            raise ValueError('[save]: data_list is not a valid list.')

    def split(self, data_list, splitting_values, ref_data_name):
        """
        Split data into multiple sub-dataset, divided accordingly to the different throttle profiles.
        :param data_list: the list of data to split.
        :param splitting_values: the list of values at which to cut the dataset.
        :param ref_data_name: the name of the data which contains the splitting_values (must be stored in the class
        object). In case the reference data contains multiple times the splitting_values, the first of them bigger than
        the previous splitting value is considered.
        """
        self.helpers.check_if_list_or_string(data_list)
        ref_data_type = self.helpers.check_if_list_or_string(ref_data_name)
        value_index_prev = 0

        if ref_data_type == 'list':
            raise ValueError(f'[split]: {ref_data_name} must be a string.')

        if isinstance(splitting_values, list):

            # size_check is used to avoid problems when creating the final dataset, in case the last splitting value is
            # repeated in the splitting_values list
            size_check = 0

            for spl_value in splitting_values:
                if isinstance(spl_value, (int, float)):
                    closest_value = min(self.data[ref_data_name], key=lambda x: abs(x - spl_value))
                    value_index = self.__search_index(ref_data_name, closest_value, value_index_prev)

                    # for the moment we assume that all data have the same size. So, the index of the ref_data_name can
                    # be used to split all data and create the dataset
                    dataset = self.__create_dataset(self.data, data_list, value_index_prev, value_index)
                    if self.debug:
                        self.logger.debug(f'[split]: created dataset dataset_{self.counter}.')
                    self.datasets.update({f"dataset_{self.counter}": dataset})
                    self.counter = self.counter + 1
                    value_index_prev = value_index + 1
                    size_check = size_check + 1

                    # we need to create also the final dataset, from value_index_prev to end of the array
                    if spl_value == splitting_values[-1] and size_check == len(splitting_values):
                        value_index_end = self.data[ref_data_name].size - 1
                        dataset = self.__create_dataset(self.data, data_list, value_index_prev, value_index_end)
                        if self.debug:
                            self.logger.debug(f'[split]: created dataset dataset_{self.counter}.')
                        self.datasets.update({f"dataset_{self.counter}": dataset})
                        self.counter = self.counter + 1
                else:
                    raise ValueError(f'[split]: splitting value {spl_value} not valid!')

        elif isinstance(splitting_values, (int, float)):
            closest_value = min(self.data[ref_data_name], key=lambda x: abs(x - splitting_values))
            value_index = self.__search_index(ref_data_name, closest_value, value_index_prev)

            # for the moment we assume that all data have the same size. So, the index of the ref_data_name can
            # be used to split all data and create the dataset
            dataset = self.__create_dataset(self.data, data_list, value_index_prev, value_index)
            if self.debug:
                self.logger.debug(f'[split]: created dataset dataset_{self.counter}.')
            self.datasets.update({f"dataset_{self.counter}": dataset})
            self.counter = self.counter + 1

            # we need to create also the final dataset, from value_index+1 to end of the array
            value_index_end = self.data[ref_data_name].size - 1
            dataset = self.__create_dataset(self.data, data_list, value_index + 1, value_index_end)
            if self.debug:
                self.logger.debug(f'[split]: created dataset dataset_{self.counter}.')
            self.datasets.update({f"dataset_{self.counter}": dataset})
            self.counter = self.counter + 1
        else:
            raise ValueError(f'[split]: splitting value {splitting_values} not valid!')

    def __create_dataset(self, data, data_list, value_index_prev, value_index):

        data_type = self.helpers.check_if_list_or_string(data_list)
        dataset = {}

        if data_type == 'str':
            # dataset composed by only one data
            dataset.update({data_list: self.data[data_list][value_index_prev:value_index]})
        elif data_type == 'list':
            for data_name in data_list:
                # dataset composed by multiple data
                dataset.update({data_name: self.data[data_name][value_index_prev:value_index]})

        return dataset

    def __search_index(self, ref_data_name, closest_value, value_index_prev):

        # locate all indexes in the array corresponding to the closest_value
        value_indexes = numpy.where(self.data[ref_data_name] == closest_value)
        value_index = 0

        for index in value_indexes[0]:
            if index > value_index_prev:
                value_index = index
                break

        if value_index == 0:
            raise ValueError(f"[search_index]: did not find a final index bigger than the initial "
                             f"index {value_index_prev}.")
        else:
            return value_index
