import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, mutual_1D, mutual_multi, network_generate, normalization_x

import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import time 
import os
import pandas as pd
import multiprocessing as mp
from scipy.optimize import fsolve, root
import networkx as nx
import scipy.integrate as sin
import seaborn as sns
import sympy as sp

fs = 18
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1
cpu_number = 12

def transition_ratio(dynamics, des, xs_low, xs_high, control_seed, control_num, control_value, t, N, parameters):
    """TODO: Docstring for transition_ratio.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :: TODO
    :returns: TODO

    """
    #random control 

    x_start = np.copy(xs_low)
    control_node = np.random.RandomState(control_seed).choice(N, control_num)
    x_start[control_node] = control_value
    ratio1 = control_num/N
    stop=1
    while stop:
        x_final = odeint(dynamics, x_start, t, args=parameters)[-1]
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
        ratio_df = pd.DataFrame(np.hstack((control_seed, ratio1)).reshape(1, 2))
        ratio_df.to_csv(des + '.csv', mode='a', index=False, header=False)
    return None

def ratio_distribution(dynamics, network_type, N, arguments, beta, control_num, control_value, control_seed_list, network_seed=0, d=None, t=np.arange(0,50,0.01)):
    """TODO: Docstring for ratio_stabilize.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, network_seed, d)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
    parameters = (N, index_i, index_j, A_interaction, cum_index, arguments)
    des = '../data/' +  dynamics.__name__ + network_type + f'N={N}_control_num={control_num}_value={control_value}'
    p = mp.Pool(cpu_number)
    p.starmap_async(transition_ratio, [(dynamics, des, xs_low, xs_high, control_seed, control_num, control_value, t, N, parameters) for control_seed in control_seed_list]).get()
    p.close()
    p.join()
    return None


arguments = (B, C, D, E, H, K)
dynamics = mutual_multi
network_type = '2D'
N = 100
beta = 1
control_value_list = [2, 3, 4, 5]
control_num_list = [10]
control_seed_list = np.arange(0, 3000, 1)

'''
for control_num in control_num_list:
    for control_value in control_value_list:
        t1 = time.time()
        ratio_distribution(dynamics, network_type, N, arguments, beta, control_num, control_value, control_seed_list)
        t2 = time.time()
        print(t2 -t1)
'''
