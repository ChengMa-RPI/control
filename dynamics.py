import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, mutual_1D, mutual_multi, network_generate, normalization_x, ode_Cheng, gif

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

fs = 18
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1

cpu_number = 10

def transition_ratio(dynamics, ratio_des, ratio_save, xs_low, xs_high, control_seed, control_num, control_value, t, N, parameters, evolution_des, evolution_save):
    """TODO: Docstring for transition_ratio.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :: TODO
    :returns: TODO

    """
    #random control 

    x_start = np.copy(xs_low)
    control_node = np.random.RandomState(control_seed).choice(N, control_num, replace=False)
    x_start[control_node] = control_value
    ratio1 = control_num/N
    stop=1
    if evolution_save:
        evolution_file = evolution_des+ f'_control_seed={control_seed}.npy'
    else:
        evolution_file = evolution_des + 'null.py'
    path = pathlib.Path(evolution_file)
    with path.open('ab') as f:
        while stop:
            x = ode_Cheng(dynamics, x_start, t, *(parameters))
            #x = odeint(dynamics, x_start, t, parameters)
            if evolution_save:
                R = normalization_x(x, xs_low, xs_high)
                np.save(f, R)

            x_final = x[-1]
            R = normalization_x(x_final, xs_low, xs_high)
            frequency, bins = np.histogram(R, bins=np.linspace(min(0, np.min(R)), max(1, np.max(R)), 6))
            num_low, num_high = frequency[0], frequency[-1]
            """
            low_boundary, high_boundary = bins[0:2], bins[-2:]
            low_node = np.where((R>=low_boundary[0]) &(R<=low_boundary[1]))[0]
            high_node = np.where((R>=high_boundary[0]) &(R<=high_boundary[1]))[0]
            """
            ratio2 = num_high/N
            stop = ratio1!=ratio2
            ratio1 = ratio2
            x_start = x_final
    if num_low + num_high < N:
        print(num_low, num_high, N, control_seed)

    else:
        if ratio_save:
            ratio_df = pd.DataFrame(np.hstack((control_seed, ratio1)).reshape(1, 2))
            ratio_df.to_csv(ratio_des + '.csv', mode='a', index=False, header=False)
    return None

def ratio_distribution(dynamics, network_type, N, arguments, beta, control_num, control_value, control_seed_list, network_seed=0, d=None, t=np.arange(0, 50, 0.01), ratio_save=1, evolution_save=0):
    """TODO: Docstring for ratio_stabilize.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, network_seed, d)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
    if network_type == '2D':
        ratio_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/beta={beta}/'
    else:
        ratio_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/d={d}/beta={beta}/'

    evolution_dir = ratio_dir + 'evolution/'
    if not os.path.exists(ratio_dir):
        os.makedirs(ratio_dir)
    if not os.path.exists(evolution_dir):
        os.makedirs(evolution_dir)

    if network_type == '2D':
        ratio_des = ratio_dir + f'N={N}_control_num={control_num}_value={control_value}'
        evolution_des = evolution_dir + f'N={N}_control_num={control_num}_value={control_value}'
    else:
        ratio_des = ratio_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}'
        evolution_des = evolution_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}'


    if ratio_save and os.path.exists(ratio_des + '.csv') and control_seed_list[0] == 0:
        print('already exists!', f'N={N}', f'control_num={control_num}', f'control_value={control_value}')
        return None

    N = np.size(A, 0)
    parameters = (N, index_i, index_j, A_interaction, cum_index, arguments)
    p = mp.Pool(cpu_number)
    p.starmap_async(transition_ratio, [(dynamics, ratio_des, ratio_save, xs_low, xs_high, control_seed, control_num, control_value, t, N, parameters, evolution_des, evolution_save) for control_seed in control_seed_list]).get()
    p.close()
    p.join()
    return None

def neighbor_dynamics(x, t, k_L, k_H, x_L, x_H, arguments):
    """TODO: Docstring for neighbor_dynamics.

    :arg1: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + k_L * x*x_L / (D + E*x + H*x_L) + k_H * x*x_H/(D + E*x + H*x_H) 
    return dxdt

def neighbor_effect(dynamics, network_type, N, arguments, beta_set, degree=4, network_seed=0, d=None, t=np.arange(0, 50, 0.01)):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    neighbor_H_set = []
    for beta in beta_set:
        for N_L in np.arange(0, degree)[::-1]:
            
            N_H = degree - N_L
            A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, network_seed, d)
            xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments, d=d)
            x_L = xs_low[0]
            x_H = xs_high[0]
            k_L = N_L/(N_L + N_H) * np.sum(A, 0)[0]
            k_H = N_H/(N_L + N_H) * np.sum(A, 0)[0]
            x = odeint(neighbor_dynamics, x_L, t, args=(k_L, k_H, x_L, x_H, arguments))[-1]
            R = (x - x_L)/(x_H - x_L)
            if R>0.2:
                neighbor_H = N_H
                break
            else:
                neighbor_H = degree + 1
        neighbor_H_set.append(neighbor_H)

    return np.array(neighbor_H_set)



arguments = (B, C, D, E, H, K)
dynamics = mutual_multi
network_type = '2D'
network_type = 'ER'
network_seed = 0
N = 900
beta_list = np.setdiff1d(np.round(np.arange(0.42, 1.3, 0.02), 2), np.round(np.arange(0.4, 1.3, 0.1), 2))
beta_list = [1]
d_list = [7200]

control_value_list = [1]
control_num_list = [100]
control_seed_list = np.arange(0, 1000, 1)
ratio_save = 1
evolution_save = 0

for d in d_list:
    for beta in beta_list:
        for control_num in control_num_list:
            for control_value in control_value_list:
                t1 = time.time()
                ratio_distribution(dynamics, network_type, N, arguments, beta, control_num, control_value, control_seed_list, network_seed=network_seed, d=d, ratio_save=ratio_save, evolution_save=evolution_save)
                t2 = time.time()
                print(t2 -t1)
'''
beta_set = np.arange(0, 2, 0.1)
neighbor_H = neighbor_effect(dynamics, network_type, N, arguments, beta_set)
'''
