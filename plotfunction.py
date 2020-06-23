import numpy as np
import matplotlib as mpl
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from operator import itemgetter
import os
from scipy.spatial import distance

# Path of file
data_points_file = "dataset.txt"
front_file = "approximation_set_final.dat"

def file_to_nparray(filepath: str):
    f = open(filepath, 'r')
    res_array = []
    for line in f:
        content = line
        if content[0] == " ":
            content = content[1:]
        if content[-1] == " ":
            content = content[:-1]
        content = ' '.join(content.split())
        content = content.split(" ")

        res_array.append(content)
    return np.array(res_array, dtype=float)

def data_points(file: str):
    """
    Loads the data points from the original file
    :param file: filepath
    :return:
    """
    f = open(file, 'r')
    for i, line in enumerate(f):
        content = line.split("];")[0].split("[")[-1]
        content = content if content[0] != " " else content[1:]
        if i == 2:
            x = np.fromstring(content, dtype=float, sep=' ')
        if i == 3:
            y = np.fromstring(content, dtype=float, sep=' ')
        if i == 3:
            y_without_noise = np.fromstring(content, dtype=float, sep=' ')
    return x, y, y_without_noise


def radial_basis_function(x_input, func_parameters):
    """
    Returns the output of the radial basis function
    f = sum over k w_k * exp(-(x-mu_k)^2 / (2* sigma_k^2))
    :param x_input: the input (np array)
    :param func_parameters: an array of (w_k, mu_k, sigma_k)
    :return: the output of f(x)
    """
    k = 10
    rad = [[] for _ in range(k)]
    for j in range(k):
        for i, x in enumerate(x_input):
            rad[j].append(func_parameters[0][j] * np.exp((-(pow(x - func_parameters[1][j],2))) / (2*pow(func_parameters[2][j], 2))))
        plt.scatter(x_input, rad[j], s=1)
    gauss_sum = np.sum(rad,axis=0)
    plt.scatter(x_input, rad[j], s=1, c="b", marker='o')
    plt.scatter(x_input, gauss_sum, s=3, c="r", marker="x")
    plt.scatter(x_input, true_function(x_input), s=3, c="g", marker="x")
    plt.show()
    return gauss_sum

def confirm_claimed_fitness(data, parameters):
    res_0 = np.subtract(np.array(data[1]), radial_basis_function(np.array(data[0]), parameters))
    f_0 = np.sum(np.square(res_0))/len(data[0])
    f_1 = np.sum((np.array([value[0] * value[0] for value in parameters]))) / len(parameters)

    print("Error found:")
    print((f_0, f_1))

def true_function(x_input):
    """
    Returns the output of the true function
    y = x^3 + x - 0.5
    :param x_input: Input vector (np array)
    :return: The output of y(x)
    """
    return np.add(np.add(np.power(x_input, 3), x_input), np.full_like(x_input, 0.5))


def plot_results(x_input, datapoints, func_parameters, index, par_set):
    plt.clf()
    x_true = x_input
    y_true = true_function(x_input)
    x_data = datapoints[0] if datapoints else []
    y_data = datapoints[1] if datapoints else []
    y_radial = radial_basis_function(x_input, func_parameters)
    x_lbls = "$x$"
    y_lbls = "$y$"
    titles = str(par_set)+" "+str(index)
    legend = []

    fig, axes = plt.subplots(nrows=1, ncols=1)
    # Plot Graph
    axes.plot(x_true, y_true, linewidth=1, c='b')
    legend.append("y(x)")

    if y_radial.size != 0:
        axes.plot(x_true, y_radial, c='r')
        legend.append(r'$r(x|\theta)$')

    if y_data.size != 0:
        axes.scatter(x_data, y_data, linewidth=1, c='green')
        legend.append("y(x) + N(0,5)")

    axes.legend(legend, loc='best')


    axes.set_xlabel(x_lbls)
    axes.set_ylabel(y_lbls)
    axes.grid()

    plt.title(titles, fontsize=16)
    plt.show()
    plt.savefig("plots/"+str(index)+"_pop"+str(par_set[2])+"appset_"+str(par_set[4])+".png")


def plot_front_and_get_param(front_file, min_index, white):
    dist = 1000

    z = (0,0)
    data_array = file_to_nparray(front_file)
    data = {"params":[], "f0":[], "f1":[]}
    for i, row in enumerate(data_array):
        data["params"].append(row[0:30])
        data["f0"].append(row[30])
        data["f1"].append(row[31])

    #normalize point
    f0_min=min(data["f0"])
    f0_max=max(data["f0"])
    norm_f0 = [(x-f0_min)/(f0_max-f0_min) for x in data["f0"]]
    f1_min=min(data["f1"])
    f1_max=max(data["f1"])
    norm_f1 = [(x-f1_min)/(f1_max-f1_min) for x in data["f1"]]
    for i, row in enumerate(data_array):
        sol = (norm_f0[i],norm_f1[i])
        d_i = distance.euclidean(z, sol)
        if d_i < dist:
            dist=d_i
            best_index = i
    #Plot Pareto Front
    if white:
        plt.scatter(data["f0"], data["f1"], marker='x')
    else:
        plt.scatter(data["f0"], data["f1"], marker='o')
    plt.scatter(data["f0"][best_index], data["f1"][best_index], c='r')
    #opt_params = data["params"][best_index]
    #param = [[],[],[]]
    #j=0
    #while j<len(opt_params):
    #    print(j, j+1, j+2)
    #    param[0].append(opt_params[j])
    #    param[1].append(opt_params[j+1])
    #    param[2].append(opt_params[j+2])
    #    j+=3
    #print(param)
    #return param

def plot_rbf(x, param):
    for i in range(0,10):
        print(param[1][i], param[2][i])
        plt.plot(x, norm.pdf(x, param[1][i], parameters[2][i]))
        plt.show()
        

if __name__ == "__main__":
    pro = 60        #: Multi-objective optimization problem index (minimization).
    k = 10          #: Number of radial basis function.
    dim = k*3       #: Number of parameters (if the problem is configurable).
    low = 0         #: Overall initialization lower bound.
    upp = 50        #: Overall initialization upper bound.
    pop = 500       #: Population size.
    ela = 500       #: Max Elitist archive size target.
    apprs = 10      #: Approximation set size (reduces size of ela after optimization using gHSS.
    eva = 200000    #: Maximum number of evaluations of the multi-objective problem allowed.
    sec = 600       #: Time limit in seconds.
    vtr = 0         #: The value to reach. If the hypervolume of the best feasible solution reaches this value, termination is enforced (if -r is specified).
    varm = 10       #: Variance multiplier for WhiteBox RBF optimization.
    wrp = "./"      #: write path.

    x = np.linspace(-3,2, 1000)
    rnd = random.randint(1,2000)
    par_set = [low,upp,pop,ela,apprs,rnd]
    os.system("rm *.dat")
    command = "./mamalgam "+str(pro)+" "+str(k)+" "+str(dim)+" "+ str(low) + " " + str(upp)+ " " + str(pop) +" "+str(ela)+" "+str(apprs)+" "+str(eva)+" "+str(sec)+ " " +str(vtr)+" "+str(varm)+" "+str(rnd)+" \"./\""
    print("COMMAND: "+command)
    os.system(command)
    parameters = plot_front_and_get_param(front_file,0,0)
    command = "./mamalgam -W "+str(pro)+" "+str(k)+" "+str(dim)+" "+ str(low) + " " + str(upp)+ " " + str(pop) +" "+str(ela)+" "+str(apprs)+" "+str(eva)+" "+str(sec)+ " " +str(vtr)+" "+str(varm)+" "+str(rnd)+" \"./\""
    print("COMMAND: "+command)
    os.system(command)
    parameters = plot_front_and_get_param(front_file,0,1)
    

    plt.xlabel("f0")
    plt.ylabel("f1")
    plt.legend(['blackbox',"whitebox"])
    plt.show()
    data = data_points(data_points_file)
    confirm_claimed_fitness(data, parameters)

