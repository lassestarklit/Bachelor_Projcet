import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats
import seaborn as sns

class Comparison:
    data = []
    classes = []
    def __init__(self, all_atts):
        self.all_attributes = all_atts

    def config_graph(self, target_name, class_names, independent_variable):
        self.independent_variable=independent_variable

        # Get indexes that match the group
        list_unique_names = class_names


        class_indices = []


        for unique_value in list_unique_names:
            class_indices.append(self.all_attributes.get_table().index[self.all_attributes.get_table()[target_name] == unique_value].tolist())

        # Merge the two lists to create a dictionary
        self.group_index_dict = dict(zip(list_unique_names, class_indices))

        #checks if data is independent variable is binary
        unique = np.unique(independent_variable.values)

        self.classes = []
        self.data = []

        for key, index in self.group_index_dict.items():
            self.classes.append(key)
            self.data.append(np.array(independent_variable.iloc[index]).reshape(-1))

        #Check if feature is binary
        self.binary = True
        try:
            for value in unique:


                value = float(value)

                if value != 0 and value != 1:
                    self.binary= False
                    break
        except ValueError:
            return

    def get_graph(self):
        #This is not necessary as feature cannot be binary
        if self.binary:

            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            color_pick = 0
            bin_dict = {}

            fig = Figure(figsize=(5,5), dpi=100)
            self.ax = fig.add_subplot(111)
            num = 0.0
            N = 2
            x = np.arange(N)
            for i in range(len(self.data)):
                unique, counts = np.unique(self.data[i], return_counts=True)

                bin_count = []

                bin_dict[self.classes[i]] = bin_count
                col = colors[i]
                name = self.classes[i]

                if num == 0:
                    self.ax.bar(x, counts, width=0.2, color=col, align='center', label=name)
                    num += 1
                elif num > 0:
                    self.ax.bar(x - (num * 0.2), counts, width=0.2, color=col, align='center', label=name)
                    num *= -1
                else:
                    self.ax.bar(x - (num * 0.2), counts, width=0.2, color=col, align='center', label=name)
                    num *= -1
                    num += 1

            handles, labels = self.ax.get_legend_handles_labels()
            import operator
            hl = sorted(zip(handles, labels),
                        key=operator.itemgetter(1))
            handles2, labels2 = zip(*hl)
            self.ax.legend(handles2, labels2)



            return fig
        else:
            # Boxplot if feature is not binary
            fig = Figure(figsize=(5, 5), dpi=100)
            self.ax=fig.add_subplot(111)
            self.ax.boxplot(self.data)
            self.ax.set_xticks(list(range(1, len(self.classes) + 1)))
            self.ax.set_xticklabels(self.classes)


            return fig

    def perform_ttest(self):
        tests_txt=""
        for x in range(1, len(self.classes)):
            tests_txt+=self.get_result_of_ttest(self.data[0], self.data[x], self.classes[0], self.classes[x], self.independent_variable.columns[0])
            tests_txt+="\n"
        for x in range(1, len(self.classes) - 1):

            j = 1
            k = x + 1
            g = 1

            while j < len(self.classes) - 1:

                if k > len(self.classes) - 1:

                    tests_txt+=self.get_result_of_ttest(self.data[x], self.data[g], self.classes[x], self.classes[g], self.independent_variable.columns[0])
                    tests_txt += "\n"
                    g += 1

                else:
                    tests_txt += self.get_result_of_ttest(self.data[x], self.data[k], self.classes[x], self.classes[k], self.independent_variable.columns[0])
                    tests_txt += "\n"
                    k += 1
                j += 1
        return tests_txt

    def get_result_of_ttest(self, data_a, data_b, name_a, name_b, variable_name):
        [tstatistic, pvalue] = stats.ttest_ind(data_a, data_b,equal_var=False)
        if pvalue < 0.05:
            return('-Pvalue implies a difference between {0} and {1} from the feature {2}'.format(name_a, name_b, variable_name))
        else:
            return('-No difference between {0} and {1} from the feature {2}'.format(name_a, name_b, variable_name))

    def plot_correlation_matrix(self, frame):

        fig, ax = plt.subplots(figsize=(11, 9))


        # calculate the correlation matrix
        corr = frame.get_corr_for_heatmap()


        # plot the heatmap
        sns.heatmap(corr,
                    xticklabels=corr.columns,
                    yticklabels=corr.columns)

        return fig




