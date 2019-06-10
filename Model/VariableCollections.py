import pandas as pd
import numpy as np
from sklearn import preprocessing
import ntpath


class TableOfVariables:
    table = pd.DataFrame()

    def get_table(self):
        return self.table

    def get_variable_names(self):
        return self.table.columns

    def get_number_of_variables(self):
        return len(self.table.columns)

    def get_variable_at_index(self, index):
        return self.table.iloc[:, [index]]

    def get_variable_name_at_index(self, index):
        return self.table.columns.values[index]


class AllAttributes(TableOfVariables):
    file_name: str

    def __init__(self, table, path):
        self.table = table
        self.table.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

        self.file_name = ntpath.basename(path)

    def variable_is_numerical(self, variable):
        unique = np.unique(self.table[variable])
        try:
            for each in unique:
                float(each)
            return True
        except ValueError:
            return False

    def variable_is_binary(self, variable):
        unique = np.unique(self.table[variable])

        if len(unique) == 2:
            return True
        else:
            return False

    def get_file_name(self):
        return self.file_name

    def get_variable(self, name):
        return self.table[name]

    def get_unique_classes_of_variable(self, variable_name):
        return np.unique(self.table[variable_name])


class EditableTables(TableOfVariables):

    def add(self, column):
        self.table = pd.concat([self.table, column], axis=1, sort=False)

    def delete(self, column):
        self.table.drop(columns=[column], inplace=True)

    def delete_all(self):
        self.table = pd.DataFrame()


class AllAvailable(EditableTables):
    def __init__(self, table):
        self.table = table


class DependentVariables(EditableTables):
    list_of_classes = []

    def add(self, column):
        self.add_classes_to_list(column)
        EditableTables.add(self, column)

    def delete(self, column_name):
        self.remove_classes_from_list(column_name)
        EditableTables.delete(self, column_name)

    def delete_all(self):
        EditableTables.delete_all(self)

        self.list_of_classes = []

    def add_classes_to_list(self, column):
        # create temp list to insert in dependent group
        unique_classes = np.unique(column)

        temp = []
        for variable_class in unique_classes:
            temp.append(variable_class)
        self.list_of_classes.append(temp)

    def remove_classes_from_list(self, attribute_name):
        for i in range(len(self.table) - 1):
            if (self.get_variable_name_at_index(i) == attribute_name):
                self.list_of_classes.pop(i)
                break

    def get_classes_of_variable_at_index(self, index):
        return self.list_of_classes[index]


class IndependentVariables(EditableTables):
    rescaled_variables = pd.DataFrame()
    active_variables = pd.DataFrame()

    def add(self, column):
        EditableTables.add(self, column)
        try:
            self.add_to_rescaled_table(column)
        except ValueError:
            self.active_variables()

    def delete(self, column_name):
        EditableTables.delete(self, column_name)
        self.remove_from_rescaled_table(column_name)

    def delete_all(self):
        EditableTables.delete_all(self)

        self.rescaled_variables = pd.DataFrame()
        self.active_variables = pd.DataFrame()

    def add_to_rescaled_table(self, column):

        x = column.values.astype(float)

        #  minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()

        # object to transform the data to fit minmax processor
        x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))

        # Run the normalizer on the dataframe
        temp = pd.DataFrame(x_scaled)
        self.rescaled_variables = pd.concat([self.rescaled_variables, temp], axis=1, sort=False)

        self.rescaled_variables.rename(columns={0: column.name}, inplace=True)

        # update active list
        self.update_active_table()

    def remove_from_rescaled_table(self, column_name):
        try:
            self.rescaled_variables.drop(columns=[column_name], inplace=True)
        except KeyError:
            pass
            # instantiate active list
        self.update_active_table()

    def variable_is_rescaled(self, col_name):

        return col_name in self.rescaled_variables

    def get_rescaled_table(self):
        return self.rescaled_variables

    def rescale_single_value(self, col_name, val):
        x = self.table[[col_name]].values.astype(float)
        #  minimum and maximum processor object
        min_max_scaler = preprocessing.MinMaxScaler()

        min_max_scaler.fit(x)
        return min_max_scaler.transform([[val]])[0][0]

    def update_active_table(self):
        self.active_variables = self.rescaled_variables.combine_first(self.table)

    def get_active_table(self):
        return self.active_variables

    def get_active_variable_at_index(self, index):
        return self.active_variables.iloc[:, [index]]

    def get_active_variable_name_at_index(self, index):
        return self.active_variables.columns.values[index]

    def get_active_variable_value_at(self, i, j):
        return self.active_variables.iloc[i, j]

    def get_corr_for_heatmap(self):
        return self.table.corr()

    def get_variable_mean(self, col_name):
        return self.table[col_name].mean()

    def get_variable_max_min(self, col_name):
        return self.table[col_name].max(), self.table[col_name].min()


class TargetVariable(EditableTables):
    encoder = preprocessing.LabelEncoder()

    def label_encode_variable(self, variable):

        self.encoder.fit(self.table[variable])

        self.table['variable' + '_encoded'] = \
            self.encoder.transform(self.table[variable])

    def get_target_variable(self):
        # if the variable is encoded the frame has 2 columns
        if len(self.table.columns) == 2:
            return self.get_variable_at_index(1)
        else:
            return self.get_variable_at_index(0)

    def get_labels(self):
        # if the variable is encoded the frame has 2 columns
        unique = np.unique(self.get_variable_at_index(0))
        if len(self.table.columns) == 2:

            labels = []
            for i in range(len(unique)):
                labels.append(self.encoder.inverse_transform([i])[0])
        else:
            labels = unique

        return labels

    def get_feature_correlation(self, feature):
        encoded = self.table.apply(preprocessing.LabelEncoder().fit_transform)
        cor_matrix = pd.concat([feature, encoded], axis=1, sort=False)
        return cor_matrix.corr().iloc[0, 1]
