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

cpu_number = 10

def mutual_constant(x, t, N, index_i, index_j, A_interaction, cum_index, arguments, control_node, control_value):
    """TODO: Docstring for mutual_constant.

    :x: TODO
    :t: TODO
    :N: TODO
    :index_i: TODO
    :index_j: TODO
    :A_interaction: TODO
    :cum_index: TODO
    :arguments: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    x[control_node] = control_value  # set the state value of control nodes constant.
    f = B + x * (1 - x/K) * ( x/C - 1)
    g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    f[control_node] = 0
    sum_g = np.add.reduceat(g, cum_index[:-1])
    sum_g[control_node] = 0

    dxdt = f + x * sum_g

    return dxdt
 
def transition_ratio(dynamics, ratio_des, ratio_save, xs_low, xs_high, control_seed, control_num, control_value, t, N, parameters, evolution_des, evolution_save):
    """TODO: Docstring for transition_ratio.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :: TODO
    :returns: TODO

    """

    random_state = np.random.RandomState(control_seed)
    control_node = random_state.choice(N, int(control_num*N), replace=False)
    x_start = np.copy(xs_low)
    parameters = parameters + (control_node, control_value)
    state1 = np.sum(x_start)
    stop=1
    if evolution_save:
        evolution_file = evolution_des+ f'_control_seed={control_seed}.npy'
    else:
        evolution_file = evolution_des + 'null.npy'
    path = pathlib.Path(evolution_file)
    with path.open('ab') as f:
        while stop:
            x = ode_Cheng(dynamics, x_start, t, *(parameters))
            if evolution_save:
                np.save(f, x)

            x_final = x[-1]
            state2 = np.sum(x_final)
            stop = np.heaviside(state2- state1 > 0.1, 0)
            state1 = state2
            

    if ratio_save:
        data = np.hstack((control_seed, x_final))
        ratio_df = pd.DataFrame(data.reshape(1, len(data)))
        ratio_df.to_csv(ratio_des + '.csv', mode='a', index=False, header=False)
    return None

def ratio_distribution(dynamics, network_type, N, arguments, beta, betaeffect, control_num, control_value, control_seed_list, network_seed=0, d=None, t=np.arange(0, 50, 0.01), ratio_save=1, evolution_save=0):
    """TODO: Docstring for ratio_stabilize.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
    average_degree = int(2 * d/N)
    ratio_dir = '../data/' +  dynamics.__name__ + '/' + network_type + f'/c={average_degree}/'
    if betaeffect:
        ratio_dir = ratio_dir + f'/beta={beta}/value={control_value}/'
    else:
        ratio_dir = ratio_dir + f'/edgewt={beta}/value={control_value}/'


    evolution_dir = ratio_dir + 'evolution/'
    if not os.path.exists(ratio_dir):
        os.makedirs(ratio_dir)
    if not os.path.exists(evolution_dir):
        os.makedirs(evolution_dir)

    ratio_des = ratio_dir + f'netseed={network_seed}_N={N}_control_num={control_num}'
    evolution_des = evolution_dir + f'netseed={network_seed}_N={N}_control_num={control_num}'


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



arguments = (B, C, D, E, H, K)
dynamics = mutual_constant
network_type = '2D'
network_type = 'ER'
network_seed = 0
N = 1000
beta_list = np.setdiff1d(np.round(np.arange(0.42, 1.3, 0.02), 2), np.round(np.arange(0.4, 1.3, 0.1), 2))
beta_list = [1]
beta_list = [0.1]
average_degree = np.array([4, 8, 16])
average_degree = np.array([16])
d_list = np.array(N * average_degree /2, dtype=int)

control_value_list = [5]
control_num_list = np.setdiff1d(np.round(np.arange(0.26, 1, 0.01), 2), np.round(np.arange(0.3, 1, 0.1), 1))
control_num_list = [0.25]


control_seed_list = [0]
control_seed_list = np.arange(0, 100, 1)
ratio_save = 1
evolution_save = 0
betaeffect = 0

for d in d_list:
    for beta in beta_list:
        for control_num in control_num_list:
            for control_value in control_value_list:
                t1 = time.time()
                ratio_distribution(dynamics, network_type, N, arguments, beta, betaeffect, control_num, control_value, control_seed_list, network_seed=network_seed, d=d, ratio_save=ratio_save, evolution_save=evolution_save)
                t2 = time.time()
                print(t2 -t1)

egwt_list = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
egwt_list = np.array([0.1])
degree_list = np.arange(0, 100, 1)
#neighbor_H = neighbor_effect(dynamics, network_type, N, network_seed, d_list[0], betaeffect, arguments, egwt_list, degree_list)
#neighbor_H = threshold_neighbor(dynamics, network_type, N, network_seed, d_list[0], betaeffect, arguments, egwt_list, degree_list)
