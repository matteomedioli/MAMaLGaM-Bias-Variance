# Written by Damy Ha
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import itemgetter

# Parameters
# path_exp : The path of your experiment folder
# y_axis_column : Hypervolume or Hypervolume improvement
# x_axis_column : Partial evaluations or real evaluations
# Colors : An array containing the colors of your line, make sure you have enough

# This script expects the results to be structured as:
# path_exp\algorithm_name\folder_of_a_single_run\statistics.dat
# Expected name of "folder_of_a_single_run" :
# experiment_name="$Algorithm"_problem"$problem_number"_size"$problem size"_pop"$population size"_evaluations"$max_evals"_run"$run"_"$subrun"
#
# Example of file structure:
# "\MO-RV-GOMEA\experiments_convergence" : path_exp
# - "MORVGOMEA_Blackbox_F1" : algorithm 1
# -- "MORVGOMEA_Blackbox_F1_problem10_size30_pop-1_evaluations10000000_run0_0" : experiments folder of run 0
# --- "statistics.dat"  : file containing the data
# --- ...
# -- "MORVGOMEA_Blackbox_F1_problem10_size30_pop-1_evaluations10000000_run1_0" : experiments folder of run 0
# --- "statistics.dat"
# --- ...

# An example of the structure of statistics.dat is given below.
# If you have a different structure you might want to change "retrieve_data_from_statistics_file()"
# Example of statistics.dat (including first comment):
## Generation  Evaluations  Time (s)  Best_obj[0]  Best_obj[1]  Hypervolume(approx. set)  [ Pop.index  Subgen.  Pop.size  ]  #Real.Evals
#           0         732       0.015   1.898e+02   7.711e-02   0.000e+00  [    0      0        732 ]          732
#           1       29259       0.631   2.675e+01   1.180e-05   1.425e+04  [    0     29        732 ]        29259
# ...


# Parameters
path_root = os.path.dirname(os.path.realpath(__file__))
path_exp = str(path_root) + "\MO-RV-GOMEA\experiments_convergence"

y_axis_column = 0  # 0: Hypervolume, 1: Hypervolume improvement
x_axis_column = 0  # 0: Evaluations, 1: Real.Evals

colors_graph = [('c', 'b'), ('lightgreen', 'green'), ('orange', 'r'), ('m', 'pink')] # The colors of your lines (STD, MEAN)


def retrieve_experiment_parameters(foldername: str):
    """
    Retrieve experiment parameters from foldername
    fodername preferable should only be the name of the folder
    :param foldername: (string) name of the folder that contains statistics.dat
    :return: dictionary containing the information (algorithm, problem_number, problem_size, pop_size, evaluations, research_run, success_run)
    """
    results_dic = {}
    foldername = foldername.split("/")[-1] if "/" in foldername or "/" in foldername else foldername
    results_dic["algorithm"] = foldername.split("_problem")[0]
    results_dic["problem_number"] = int(foldername.split("_problem")[-1].split("_")[0])
    results_dic["problem_size"] = int(foldername.split("_size")[-1].split("_")[0])
    results_dic["pop_size"] = int(foldername.split("_pop")[-1].split("_")[0])
    results_dic["evaluations"] = int(foldername.split("_evaluations")[-1].split("_")[0])
    results_dic["research_run"] = int(foldername.split("_run")[-1].split("_")[0])
    results_dic["success_run"] = int(foldername.split("_run")[-1].split("_")[1])
    return results_dic


# Retrieve fitness and evaluations (fitness, evaluations)
def retrieve_data_from_statistics_file(filepath: str, return_as_dict: bool):
    """
    Retrieve data and return as a dictionary of np.arrays() when return_as_dict = True
    Return data as array of rows [[evaluations, time, hypervolume, real evaluations]] when False
    :param filepath: the folder that contains statistics.dat
    :return: a dictionary containing the data
    """
    f = open(filepath + "/statistics.dat", "r")

    data = []
    # Read last line
    for line in f:
        if "#" not in line:
            content = line.replace("[", "")
            content = content.replace("]", "")
            content = ' '.join(content.split()).split(' ')
            if not return_as_dict:
                data.append([int(content[1]), float(content[2]), float(content[5]), int(content[-1])])
            else:
                data.append(content)

    if not return_as_dict:
        return data

    data_flipped = list(map(list, zip(*data)))
    return {"generation": np.array(data_flipped[0], dtype=int),
            "evaluations": np.array(data_flipped[1], dtype=int),
            "time": np.array(data_flipped[2], dtype=float),
            "hypervolume": np.array(data_flipped[5], dtype=float),
            "real_evaluations": np.array(data_flipped[-1], dtype=int)}


def create_graph_data(dictionary: dict):
    """
    Converts the experiment data to graph data
    :param dictionary: The input data expected as {Algorithm:[dictionary_runs]}
    :return: A restructured dictionary {Algorithm:([X], [STD_MIN], [MEAN], STD_MAX)}
    """
    results = {}
    # Loop over algorithms
    for algorithm in dictionary.keys():

        all_data = []
        # Given the algorithm, for all runs, put the lines of that run into one array
        # example of a line: [evaluations, time, hypervolume, real evaluations]
        for run in dictionary[algorithm]:
            previous_hypervolume = 0
            for i, line_of_data in enumerate(run):
                # Calculate hypervolume improvement
                if y_axis_column == 1:
                    if i == 0:
                        previous_hypervolume = line_of_data[2]
                        line_of_data[2] = 0
                    else:
                        current_hypervolume = line_of_data[2]
                        line_of_data[2] = line_of_data[2] - previous_hypervolume
                        previous_hypervolume = current_hypervolume

                # print(line_of_data[2])
                all_data.append(line_of_data)

        # Sort array based on number of evaluations (will become the x-axis)
        all_data = sorted(all_data, key=itemgetter(0))

        # Loop through all sorted data lines
        # Group the x-values (and y-values) based on significance
        eval_data = []  # x-axis an 1D array
        hypervolume_data = []  # keeps the hypervolume as [[],...,[]]
        for line in all_data:
            # Switch between line[0] (partial eval) and line[3] (real feval)
            if x_axis_column == 0:
                x_value = line[0]
            else:
                x_value = line[3]


            if x_value < 10000:
                significance = -3
            elif x_value < 100000:
                significance = -4
            elif x_value < 1000000:
                significance = -5
            elif(x_value < 10000000):
                significance = -6

            if round(x_value, significance) in eval_data:
                hypervolume_data[-1].append(line[2])
            else:
                eval_data.append(round(x_value, significance))
                hypervolume_data.append([line[2]])

        h_data_mean = []
        h_data_std = []
        # Determine mean and std
        for h_array in hypervolume_data:
            if len(h_array) == 1:
                h_data_mean.append(h_array[0])
                h_data_std.append(0.0)
            else:
                h_data_mean.append(np.mean(np.array(h_array)))
                h_data_std.append(np.std(np.array(h_array)))

        h_data_mean = np.array(h_data_mean)
        h_data_std = np.array(h_data_std)
        eval_data = np.array(eval_data, dtype=int)
        results[algorithm] = (
            eval_data, np.subtract(h_data_mean, h_data_std), h_data_mean, np.add(h_data_mean, h_data_std))

        # print(algorithm)
        # print("X:", eval_data)
        # print("mean", h_data_mean)
        # print("std", h_data_std)

    return results


def creategraph(graph_data, colors):
    """
    Creates a log-log graph of graph_data
    :param graph_data: (dict) expects the input given as {Algorithm:{([Dim size], [Min std], [Mean], [max STD]) }}
    :return:
    """
    # Individual Settings:
    # x_ticks = [1000, 10000, 100000]
    # y_ticks = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000]
    x_lbls = "Evaluations"
    y_lbls = "Hypervolume" if y_axis_column == 0 else "Hypervolume improvement"
    titles = "Hypervolume vs Evaluations"

    legend_data = []

    fig, axes = plt.subplots(nrows=1, ncols=1)
    # Plot Graph
    for i, algorithm in enumerate(graph_data):
        axes.plot(graph_data[algorithm][0], graph_data[algorithm][1], ls=':', c=colors[i][0], linewidth=2,
                  label='_nolegend_')
        axes.plot(graph_data[algorithm][0], graph_data[algorithm][2], c=colors[i][1], linewidth=2)
        axes.plot(graph_data[algorithm][0], graph_data[algorithm][3], c=colors[i][0], ls=':', linewidth=2)

        legend_data.append(algorithm + "(STD)")
        legend_data.append(algorithm + "(mean)")

    # Graph Settings
    # axes.set_xlim(x_ticks[0], x_ticks[-1])
    # axes.set_ylim(0, 4000000)
    axes.set_xscale('log')
    axes.set_yscale('log')

    # axes.set_xticks(x_ticks)
    # axes.set_yticks(y_ticks)

    axes.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes.grid()

    axes.set_xlabel(x_lbls)
    axes.set_ylabel(y_lbls)

    axes.legend(legend_data)
    fig.suptitle(titles, fontsize=16)
    plt.show()


# Start
print("Starting convergence analysis")
grouped_by_algorithm = {}  # {Algorithm:runs[data[] / data{}]}

# Loop over algorithms
for algorithm in os.listdir(path_exp):
    # Check for directory and correct experiment
    if os.path.isdir(path_exp + "/" + algorithm):
        # Set path and create new dictionary for algorithm
        path_algorithm = path_exp + "/" + algorithm

        grouped_by_algorithm[algorithm] = []
        # Loop over results folder
        for results_folder in os.listdir(path_algorithm):
            # Check for directory
            if os.path.isdir(path_algorithm + "/" + results_folder):
                # Retrieve parameters
                # (algorithm, problem_number, problem_size, pop_size, evaluations, research_run, success_run)
                params_exp = retrieve_experiment_parameters(results_folder)

                # Set path and create new dictionary for result
                path_result = path_algorithm + "/" + results_folder

                # Retrieve run data
                data_of_run = retrieve_data_from_statistics_file(path_result, False)
                grouped_by_algorithm[algorithm].append(data_of_run)

# Create graph data
data_graph = create_graph_data(grouped_by_algorithm)
# print(data_graph)

# Create graph
creategraph(data_graph, colors_graph)
#
# # Tests
# # test_path_1 = "MORVGOMEA_Blackbox_problem10_size60_pop-1_evaluations10000000_run0_0"
# # Testing correct info extracted from path
# # print(retrieve_experiment_parameters(test_path_1))
# # print(retrieve_elitist_solution_data(test_path_2))
