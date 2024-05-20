import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, network_generate, normalization_x, ode_Cheng, gif, mutual_1D
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, kcore_KNN, kcore_shell, kcore_KNN_degree, reducednet_effstate, neighborshell_given_core, group_partition_degree_interval

import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import pandas as pd
import multiprocessing as mp
from scipy.optimize import fsolve, root
import networkx as nx
import scipy.integrate as sin
import seaborn as sns
import sympy as sp
import pathlib 

fontsize = 22
ticksize = 18
legendsize = 16
lw = 2.5
alpha = 0.8

B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1

cpu_number = 4

def mutual_multi_constant(x, t, control_node, control_constant, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[control_node] = control_constant 
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    dxdt[control_node] = 0.0
    return dxdt

def random_control(network_type, N, weight, network_seed, d, control_num, control_seed, attractor_value):
    """TODO: Docstring for random_control.

    :arg1: TODO
    :returns: TODO

    """

    dynamics_multi_constant = globals()[dynamics + '_multi_constant']
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
    beta = betaspace(A, [0])[0]
    N_actual = len(A)
    random_state = np.random.RandomState(control_seed)
    control_order = random_state.choice(N_actual, N_actual, replace=False)
    node_r = control_order[:int(control_num*N_actual)]
    node_uncontrol = np.setdiff1d(control_order, node_r)

    des = '../data/' + dynamics + '/' + network_type + '/multi_dynamics/'
    if not os.path.exists(des):
        os.makedirs(des)
    "net arguments"
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    "original multi system"
    initial_condition = np.ones(N_actual) * attractor_value
    xH = odeint(globals()[dynamics + '_1D'], np.array([5]), t, args=(beta, arguments))[-1]
    xs_multi = odeint(dynamics_multi_constant, initial_condition, t, args=(node_r, xH, arguments, net_arguments))[-1]

    "data saving"
    data_xs = np.hstack((control_num, xs_multi))
    df_xs = pd.DataFrame(data_xs.reshape(1, len(data_xs)))
    des_file = des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_multi_xs.csv'
    df_xs.to_csv(des_file, mode = 'a', index=False, header=False)
    return None

def random_control_parallel(network_type, N, weight, network_seed, d, control_num, control_seed_list, attractor_value):
    """TODO: Docstring for MFT_parallel.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(random_control, [(network_type, N, weight, network_seed, d, control_num, control_seed, attractor_value) for control_seed in control_seed_list]).get()
    p.close()
    p.join()
    return None

def heatmap_dynamics(network_type, N, d, weight, network_seed, control_seed, control_num, attractor_value, plot_t):
    """TODO: Docstring for heatmap.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :weight: TODO
    :network_seed: TODO
    :control_seed: TODO
    :control_size: TODO
    :returns: TODO

    """
    dynamics_multi_constant = globals()[dynamics + '_multi_constant']
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
    G = nx.from_numpy_matrix(A)
    beta = betaspace(A, [0])[0]
    N_actual = len(A)
    random_state = np.random.RandomState(control_seed)
    control_order = random_state.choice(N_actual, N_actual, replace=False)
    node_r = control_order[:int(control_num*N_actual)]
    node_uncontrol = np.setdiff1d(control_order, node_r)

    "net arguments"
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    "original multi system"
    initial_condition = np.ones(N_actual) * attractor_value
    xH = odeint(globals()[dynamics + '_1D'], np.array([5]), t, args=(beta, arguments))[-1]
    xs_multi = odeint(dynamics_multi_constant, initial_condition, t, args=(node_r, xH, arguments, net_arguments))
    data_snap = xs_multi[int(plot_t/0.01)]
    norm_color = data_snap/5
    cmap = plt.cm.coolwarm

    nx.draw(G, pos= nx.spring_layout(G, seed=0) , cmap=cmap, vmin=0, vmax=1, node_color=norm_color, alpha=0.8)



    return xs_multi

def R_f(control_seed_list, control_num_list):
    for i, control_seed  in enumerate(control_seed_list):
        R_list = []
        for control_num in control_num_list:
            xs_multi = heatmap_dynamics(network_type, N, d, weight, network_seed, control_seed, control_num, attractor_value, plot_t)

            R_num = np.sum(xs_multi[-1]>5) / N
            R_list.append(R_num)
        plt.plot(control_num_list, R_list, color=colors[i], linewidth=lw, alpha=alpha, label=f'realization{i}')
    plt.xlabel('control size $f$', fontsize=fontsize)
    plt.ylabel('recovery size $R$', fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.95, wspace=0.25, hspace=0.25, bottom=0.20, top=0.95)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.show()
    
        

dynamics = 'mutual'
dynamics_multi = 'mutual_constant'
arguments = (B, C, D, E, H, K)

control_num_list = np.round(np.arange(0.01, 1, 0.01), 2)
control_constant = 5
attractor_value = 0.1



control_seed_list = np.arange(4)
network_type = 'ER'
N = 1000
network_seed = 0
c = 8
d = int(N*c/2)
weight = 0.1
interval = 1


for i, control_num in enumerate(control_num_list):
    #random_control_parallel(network_type, N, weight, network_seed, d, control_num, control_seed_list, attractor_value)
    pass

control_seed = 0
control_num = 0.15
N = 100
d = 400
weight = 0.1
plot_t = 150
xs_multi = heatmap_dynamics(network_type, N, d, weight, network_seed, control_seed, control_num, attractor_value, plot_t)

control_seed_list = [0, 1, 2]
control_num_list = np.arange(0, 0.3, 0.01)
colors = ['#1b9e77', '#d95f02', '#7570b3']


