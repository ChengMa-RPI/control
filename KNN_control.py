import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, network_generate, normalization_x, ode_Cheng, gif
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

def mutual_group_decouple(x, t, arguments, w, xs_group_transpose):
    """TODO: Docstring for mutual_group_decouple.

    :x: TODO
    :t: TODO
    :arguments: TODO
    :w: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + np.sum(w * x * xs_group_transpose / (D + E * x + H * xs_group_transpose), 0)
    return dxdt 

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

def partition_connect_control(num_neighbors_control, interval):
    """ partition the subgroup according to the number of neighbors in the control set."""
    min_nn = np.min(num_neighbors_control)
    max_nn = np.max(num_neighbors_control)
    subgroup_index = []
    for i in range(min_nn, max_nn+1, interval):
        j = i + interval 
        subgroup_k = [k for k, nn_k in enumerate(num_neighbors_control) if nn_k >= i and nn_k < j]
        if subgroup_k :
            subgroup_index.append(subgroup_k)
    return subgroup_index

def random_control(network_type, N, weight, network_seed, d, control_constant, control_num, control_seed, attractor_value, method, interval):
    """TODO: Docstring for random_control.

    :arg1: TODO
    :returns: TODO

    """
    dynamics_multi_constant = globals()[dynamics + '_multi_constant']
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
    k = np.sum(A>0, 0)
    w = np.sum(A, 0)
    N_actual = len(A)
    G = nx.from_numpy_array(A)
    neighbors = G.neighbors
    random_state = np.random.RandomState(control_seed)
    control_order = random_state.choice(N_actual, N_actual, replace=False)
    node_r = control_order[:int(control_num*N_actual)]
    node_uncontrol = np.setdiff1d(control_order, node_r)
    KNN = neighborshell_given_core(G, A, node_r)
    if method == 'KNN':
        des = '../data/' + dynamics + '/' + network_type + f'/KNN/'
        group_index = KNN
    elif method == 'firstNN':
        des = '../data/' + dynamics + '/' + network_type + f'/firstNN/'
        if len(KNN) > 2:
            group_index = [KNN[0], KNN[1], np.hstack(([KNN[i+2] for i in range(len(KNN)-2)]))]
        else:
            group_index = [KNN[0], KNN[1]]
    elif method == 'KNN_connect_control':
        group_index = [KNN[0]]
        KNN_connect_control = [len([j for j in neighbors(i) if j in node_r]) for i in  KNN[1] ]
        subgroup_index = partition_connect_control(KNN_connect_control, interval)
        group_index.extend([KNN[1][i] for i in subgroup_index][::-1])
        group_index.extend([KNN[i] for i in range(2, len(KNN)) if i != 1])
        des = '../data/' + dynamics + '/' + network_type + f'/KNN_subgroup_interval={interval}/'

    if not os.path.exists(des):
        os.makedirs(des)
    "net arguments"
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    t = np.arange(0, 1000, 0.01)
    "original multi system"
    initial_condition = np.ones(N_actual) * attractor_value
    xs_multi = odeint(dynamics_multi_constant, initial_condition, t, args=(node_r, control_constant, arguments, net_arguments))[-1]

    "reduction system"
    A_reduction, net_arguments_reduction, x_eff = reducednet_effstate(A, xs_multi, group_index)
    initial_condition_reduction = np.ones(len(group_index)) * attractor_value
    xs_reduction = odeint(dynamics_multi_constant, initial_condition_reduction, t, args=(0, control_constant, arguments, net_arguments_reduction))[-1]

    "node state by group decouple"
    rearange_index = np.hstack((group_index))
    length_groups = len(group_index)
    each_group_length = [len(i) for i in group_index]
    A_rearange = A[rearange_index]
    reduce_index = np.hstack((0, np.cumsum(each_group_length)))
    xs_group_transpose = xs_reduction.reshape(length_groups, 1)
    w_group = np.add.reduceat(A_rearange, reduce_index[:-1])
    w_group_uncontrol = w_group[:, node_uncontrol]
    initial_condition_uncontrol = np.ones(len(node_uncontrol)) * attractor_value
    xs_group_decouple = np.ones(N_actual) * control_constant
    xs_group_decouple[node_uncontrol] = odeint(mutual_group_decouple, initial_condition_uncontrol, t, args=(arguments, w_group_uncontrol, xs_group_transpose))[-1]

    "data saving"
    data_reduction = np.hstack((control_num, np.ravel(A_reduction), x_eff, xs_reduction))
    reduction_df = pd.DataFrame(data_reduction.reshape(1, len(data_reduction)))
    des_file_reduction = des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}.csv'
    reduction_df.to_csv(des_file_reduction , mode = 'a', index=False, header=False)

    node_group_mapping = np.hstack(([[i] * len(subgroup) for i, subgroup in enumerate(group_index)]))
    data_multi = np.hstack((np.tile(control_num, (4, 1)), np.vstack((rearange_index, node_group_mapping, xs_multi[rearange_index], xs_group_decouple[rearange_index]))))
    multi_df = pd.DataFrame(data_multi)
    des_file_multi = des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_multi_xs.csv'
    multi_df.to_csv(des_file_multi , mode = 'a', index=False, header=False)
    return None

def random_control_parallel(network_type, N, weight, network_seed, d, control_constant, control_num, control_seed_list, attractor_value, method, interval):
    """TODO: Docstring for MFT_parallel.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(random_control, [(network_type, N, weight, network_seed, d, control_constant, control_num, control_seed, attractor_value, method, interval) for control_seed in control_seed_list]).get()
    p.close()
    p.join()
    return None

    
        

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
c = 4
d = int(N*c/2)
weight = 0.2
interval = 1

method_list = ['KNN', 'firstNN', 'KNN_connect_control']

for i, control_num in enumerate(control_num_list):
    for method in method_list:
        #random_control_KNN(network_type, N, weight, network_seed, d, control_constant, control_num, control_seed, attractor_value, method, space, degree_interval)
        random_control_parallel(network_type, N, weight, network_seed, d, control_constant, control_num, control_seed_list, attractor_value, method, interval)
        pass
    #random_control_connectnum(network_type, N, weight, network_seed, d, control_constant, control_num, control_seed, attractor_value, interval)

