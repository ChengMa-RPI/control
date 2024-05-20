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


def transition_ratio(dynamics, ratio_des, ratio_save, xs_low, xs_high, control_seed, control_num, control_value, t, N, parameters, evolution_des, evolution_save):
    """TODO: Docstring for transition_ratio.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :: TODO
    :returns: TODO

    """
    #random control 

    random_state = np.random.RandomState(control_seed)
    initialize = random_state.choice(N, int(control_num*N), replace=False)
    x_start = np.copy(xs_low)
    x_start[initialize] = control_value
    ratio1 = control_num
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
            num_low, num_high = frequency[0], sum(frequency[1:])
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
        ratio_dir = ratio_dir + f'/beta={beta}/'
    else:
        ratio_dir = ratio_dir + f'/edgewt={beta}/'


    evolution_dir = ratio_dir + 'evolution/'
    if not os.path.exists(ratio_dir):
        os.makedirs(ratio_dir)
    if not os.path.exists(evolution_dir):
        os.makedirs(evolution_dir)

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

def global_dynamics(x, t, beta, f, k_H, k_L, arguments):
    """TODO: Docstring for neighbor_dynamics.

    :arg1: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    x_h, x_l, x_i = x 
    fx = B + x * (1 - x/K) * ( x/C - 1) 
    #gx = np.array([beta * (f* x_h*x_h / (D + E*x_h + H*x_h) + (1-f) * x_h*x_l/(D + E*x_h + H*x_l)), beta * (f* x_l*x_h / (D + E*x_l + H*x_h) + (1-f) * x_l*x_l/(D + E*x_l + H*x_l)), k_H*x_i*x_h / (D + E*x_i + H*x_h) + k_L*x_i*x_l/(D + E*x_i + H*x_l)])
    gx = np.array([beta * ((1-f)* x_h*x_h / (D + E*x_h + H*x_h) + (f) * x_h*x_l/(D + E*x_h + H*x_l)), beta * (f* x_l*x_h / (D + E*x_l + H*x_h) + (1-f) * x_l*x_l/(D + E*x_l + H*x_l)), k_H*x_i*x_h / (D + E*x_i + H*x_h) + k_L*x_i*x_l/(D + E*x_i + H*x_l)])
    dxdt = fx + gx
    return dxdt

def neighbor_effect(dynamics, network_type, N, network_seed, d, betaeffect, arguments, egwt_list, degree_list, t=np.arange(0, 50, 0.01)):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, network_seed, d)
    beta, _ = betaspace(A, [0])

    neighbor_H_list = np.zeros((np.size(egwt_list), np.size(degree_list)))
    for egwt, i in zip(egwt_list, range(np.size(egwt_list))):
        x_L = odeint(mutual_1D, 0, t, args=(beta * egwt, arguments))[-1]
        x_H = odeint(mutual_1D, 5, t, args=(beta * egwt, arguments))[-1]
        for degree, j in zip(degree_list, range(np.size(degree_list))):
            for N_L in np.arange(0, degree)[::-1]:
                
                N_H = degree - N_L
                k_L = N_L * egwt
                k_H = N_H * egwt
                #x = odeint(neighbor_dynamics, x_L, np.arange(0, 500, 0.01), args=(k_L, k_H, x_L, x_H, arguments))[-1]
                x = []
                for initial_try in range(6):
                    x0 = fsolve(neighbor_dynamics, initial_try, args=(0, k_L, k_H, x_L, x_H, arguments))
                    solution = neighbor_dynamics(x0, 0, k_L, k_H, x_L, x_H, arguments) 
                    if abs(solution) < 1e-10:
                        x.append(x0)
                x = np.min(x)
                R = (x - x_L)/(x_H - x_L)
                if R>0.5:
                    neighbor_H = N_H
                    neighbor_H_list[i, j]  = neighbor_H
                    break
                else:
                    neighbor_H = degree + 1
    for neighbor_H, egwt in zip(neighbor_H_list, egwt_list):
        index = np.where(neighbor_H>0)[0]
        plt.plot(degree_list[index], neighbor_H[index], '--', label=f'w={egwt}')
    plt.xlabel('degree $k$', fontsize=fontsize)
    plt.ylabel('$N_H$', fontsize=fontsize)
    plt.subplots_adjust(left=0.15, right=0.8, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize, loc='upper left',bbox_to_anchor=(0.9, 1.)  )


    plt.show()


    return neighbor_H_list

def threshold_neighbor(dynamics, network_type, N, network_seed, d, betaeffect, arguments, egwt_list, degree_list, t=np.arange(0, 50, 0.01)):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, network_seed, d)
    beta_unweighted, _ = betaspace(A, [0])
    average_degree = int(2 * d/N)
    neighbor_H_list = np.ones((np.size(egwt_list), np.size(degree_list))) * degree_list[-1]
    for egwt, i in zip(egwt_list, range(np.size(egwt_list))):
        beta = beta_unweighted * egwt
        for degree, j in zip(degree_list, range(np.size(degree_list))):
            for N_L in np.arange(0, degree)[::-1]:
                f = 0
                N_H = degree - N_L
                k_H = N_H * egwt
                k_L = N_L * egwt
                x = odeint(global_dynamics, np.array([5, 0, 0]), np.arange(0, 500, 0.01), args=(beta, f, k_H, k_L, arguments))[-1]
                R = (x[-1] - x[1])/(x[0] - x[1])
                if R>0.5:
                    neighbor_H = N_H
                    neighbor_H_list[i, j] = neighbor_H
                    break
                else:
                    neighbor_H = degree + 1

    for neighbor_H, egwt in zip(neighbor_H_list, egwt_list):
        threshold_dir = '../data/percolation/' + network_type + f'/c={average_degree}/'
        if betaeffect:
            threshold_dir = threshold_dir + f'beta={beta}/'
        else:
            threshold_dir = threshold_dir + f'edgewt={egwt}/'
        if not os.path.exists(threshold_dir):
            os.makedirs(threshold_dir)

        threshold_df = pd.DataFrame(np.vstack((degree_list, neighbor_H)).transpose())
        threshold_df.to_csv(threshold_dir + 'threshold.csv', index=False, header=False)
        index = np.where(neighbor_H<degree_list[-1])[0]
        plt.plot(degree_list[index], neighbor_H[index], '--', label=f'w={egwt}')
    plt.xlabel('degree $k$', fontsize=fontsize)
    plt.ylabel('$N_H$', fontsize=fontsize)
    plt.subplots_adjust(left=0.15, right=0.8, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize, loc='upper left',bbox_to_anchor=(0.9, 1.)  )

    #plt.show()

    return neighbor_H_list

def state_HL(dynamics, network_type, N, arguments, beta, betaeffect, control_num, control_value, control_seed_list, network_seed=0, d=None, t=np.arange(0, 50, 0.01), ratio_save=1, evolution_save=0 ):
    """TODO: Docstring for state_HL.

    :dynamics: TODO
    :network_type: TODO
    :: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    xs_low, xs_high = stable_state(A, A_interaction, index_i, index_j, cum_index, arguments)
    random_state = np.random.RandomState(control_seed)
    initialize = random_state.choice(N, int(control_num*N), replace=False)
    x_start = np.copy(xs_low)
    x_start[initialize] = control_value
    x = ode_Cheng(dynamics, x_start, t, *(parameters))
    x_L_mean = []
    x_H_mean = []
    x_L_max = []
    x_H_min  = []
    boundary = 4
    for i in x:
        if np.sum(i<boundary):
            x_L = i[i<boundary]
            x_L_mean.append(np.mean(x_L))
            x_L_max.append(np.max(x_L))
        else:
            x_L_mean.append(0)
            x_L_max.append(0)

        if np.sum(i>boundary):
            x_H = i[i>boundary]
            x_H_mean.append(np.mean(x_H))
            x_H_min.append(np.min(x_H))
        else:
            x_H_mean.append(0)
            x_H_min.append(0)


    


arguments = (B, C, D, E, H, K)
dynamics = mutual_multi
network_type = '2D'
network_type = 'ER'
network_seed = 0
N = 10000
beta_list = np.setdiff1d(np.round(np.arange(0.42, 1.3, 0.02), 2), np.round(np.arange(0.4, 1.3, 0.1), 2))
beta_list = [1]
beta_list = [0.2]
average_degree = np.array([4, 8, 16])
average_degree = np.array([16])
d_list = np.array(N * average_degree /2, dtype=int)

control_value_list = [1]
control_num_list = [0.05]
control_seed_list = [0]
control_seed_list = np.arange(0, 100, 1)
ratio_save = 1
evolution_save = 0
betaeffect = 0

'''
for d in d_list:
    for beta in beta_list:
        for control_num in control_num_list:
            for control_value in control_value_list:
                t1 = time.time()
                ratio_distribution(dynamics, network_type, N, arguments, beta, betaeffect, control_num, control_value, control_seed_list, network_seed=network_seed, d=d, ratio_save=ratio_save, evolution_save=evolution_save)
                t2 = time.time()
                print(t2 -t1)
'''

egwt_list = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
egwt_list = np.array([0.1])
degree_list = np.arange(0, 100, 1)
#neighbor_H = neighbor_effect(dynamics, network_type, N, network_seed, d_list[0], betaeffect, arguments, egwt_list, degree_list)
neighbor_H = threshold_neighbor(dynamics, network_type, N, network_seed, d_list[0], betaeffect, arguments, egwt_list, degree_list)
