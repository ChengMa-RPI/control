import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, mutual_1D, network_generate, normalization_x, ode_Cheng, gif

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
from scipy.special import comb

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

def binary_search(xL, xR, yL, yR, threshold):
    """TODO: Docstring for binary_search.

    :xL: TODO
    :xR: TODO
    :yL: TODO
    :yR: TODO
    :threshold: TODO
    :returns: TODO

    """
    if yL <= threshold and yR >= threshold:
        xR = (xR + xL) / 2
    elif yL > threshold :
        xL, xR = xL*2-xR, xL
    elif yR < threshold:
        xL, xR = xR, xR*2-xL
    return xL, xR

def global_dynamics(x, t, beta, k_H, k_L, arguments):
    """TODO: Docstring for neighbor_dynamics.

    :arg1: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    x_h, x_l, x_i = x 
    fx = B + x * (1 - x/K) * ( x/C - 1) 
    gx = np.array([beta * x_h*x_h / (D + E*x_h + H*x_h), beta * x_l*x_l/(D + E*x_l + H*x_l), k_H*x_i*x_h / (D + E*x_i + H*x_h) + k_L*x_i*x_l/(D + E*x_i + H*x_l)])
    dxdt = fx + gx
    return dxdt

def dynamics_xH_xL(x, t, beta, x_h, x_l, k_H, k_L, arguments):
    """TODO: Docstring for neighbor_dynamics.

    :arg1: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    fx = B + x * (1 - x/K) * ( x/C - 1) 
    gx = k_H*x*x_h / (D + E*x + H*x_h) + k_L*x*x_l/(D + E*x + H*x_l)
    dxdt = fx + gx
    return dxdt

def mutual_multi(x, t, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
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

def random_control(network_type, N, beta, betaeffect, network_seed, d, control_num, control_seed, attractor_high, attractor_low):
    """TODO: Docstring for random_control.

    :arg1: TODO
    :returns: TODO

    """
    dynamics_multi_constant = globals()[dynamics + '_multi_constant']
    des = '../data/' + dynamics + '/' + network_type + '/multi_dynamics/'
    if not os.path.exists(des):
        os.makedirs(des)

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    if betaeffect == 0:
        weight = beta 
        beta = betaspace(A, [0])[0]
        des_file = des + f'N={N}_d={d}_netseed={network_seed}_weight={weight}_controlseed={control_seed}_multi_xs.csv'
    else:
        weight = A.max()
        des_file = des + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'
    N_actual = len(A)
    random_state = np.random.RandomState(control_seed)
    control_order = random_state.choice(N_actual, N_actual, replace=False)
    node_r = control_order[:int(control_num*N_actual)]

    "original multi system"
    t = np.arange(0, 1000, 0.01)
    xH = odeint(mutual_1D, np.array([attractor_high]), t, args=(beta, arguments))[-1]
    control_constant = xH
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    initial_condition = np.ones(N_actual) * attractor_low
    xs_multi = odeint(dynamics_multi_constant, initial_condition, t, args=(node_r, control_constant, arguments, net_arguments))[-1]
    "data saving"
    data_multi = np.hstack((control_num, xs_multi))
    df = pd.DataFrame(data_multi.reshape(1, len(data_multi)))
    df.to_csv(des_file, mode = 'a', index=False, header=False)
    return None

def random_control_parallel(network_type, N, beta, betaeffect, network_seed, d, control_num, control_seed_list, attractor_high, attractor_low):
    """TODO: Docstring for MFT_parallel.

    :arg1: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(random_control, [(network_type, N, beta, betaeffect, network_seed, d, control_num, control_seed, attractor_high, attractor_low) for control_seed in control_seed_list]).get()
    p.close()
    p.join()
    return None

    
        



def threshold_sH_sL_neighbor(network_type, N, weight, network_seed, d, dynamics, arguments):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
    beta, _ = np.round(betaspace(A, [0]), 5)
    xL, xH = odeint(mutual_1D, np.array([0.1, 5]), np.arange(0, 500, 0.01), args=(beta, arguments))[-1]
    sL_list = np.arange(0, 20.05, 0.05)
    sH_list = np.arange(0, 0.41, 0.01)
    xs_list = np.zeros((len(sL_list), len(sH_list)))
    for i, sL in enumerate(sL_list):
        for j, sH in enumerate(sH_list):
            #x = odeint(global_dynamics, np.array([5, 0, 0]), np.arange(0, 500, 0.01), args=(beta, sH, sL, arguments))[-1][-1]
            x = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, sH, sL, arguments))[-1]
            xs_list[i, j] = x

    threshold_des = '../data/' + dynamics + '/threshold/' 
    threshold_des_file = threshold_des +  f'xs_sL_sH_beta={beta}.csv'
    if not os.path.exists(threshold_des):
        os.makedirs(threshold_des)

    data = np.hstack(( np.hstack((0, sL_list)).reshape(len(sL_list) + 1, 1), np.vstack((sH_list, xs_list)) ))
    threshold_df = pd.DataFrame(data)
    threshold_df.to_csv(threshold_des_file, index=False, header=False)
    return None

def threshold_sH_sL_neighbor_multi(network_type, N, beta, network_seed, d, dynamics, arguments, attractor_low, attractor_high, control_seed, control_num):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    dynamics_multi_constant = globals()[dynamics + '_multi_constant']
    dynamics_1D = globals()[dynamics + '_1D']
    "network structure from 'beta'"
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, network_seed, d)
    k = np.sum(A>0, 0)
    G = nx.from_numpy_array(A)
    neighbors = G.neighbors
    t = np.arange(0, 1000, 0.01)
    N_actual = len(A)
    beta_unweighted, _ = np.round(betaspace(A, [0]), 5)
    weight = beta / beta_unweighted

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    xs_L, xs_H = odeint(dynamics_1D, np.array([attractor_low, attractor_high]), t, args=(beta, arguments))[-1]

    initial_condition = np.ones(N_actual) * attractor_low
    control_constant = xs_H
    random_state = np.random.RandomState(control_seed)
    control_order = random_state.choice(N_actual, N_actual, replace=False)
    node_r = control_order[:int(control_num*N_actual)]
    node_uncontrol = np.setdiff1d(np.arange(N_actual), node_r)
    xs_multi = odeint(dynamics_multi_constant, initial_condition, t, args=(node_r, control_constant, arguments, net_arguments))
    recovery = np.zeros((len(node_uncontrol)))
    nH_list = np.zeros((len(node_uncontrol)))
    for i, node in enumerate(node_uncontrol):
        xs_high_index = np.where(xs_multi[:, node] > attractor_high)[0]
        node_transition_time = xs_high_index[0] if len(xs_high_index) else -1
        recovery[i] = 1 if len(xs_high_index) else 0
        xs_multi_this_moment = xs_multi[node_transition_time]
        nH_list[i] = np.sum(xs_multi_this_moment[list(neighbors(node))] > attractor_high)


    nL_list = k[node_uncontrol] - nH_list
    sH = nH_list * weight
    sL = nL_list * weight
    data = np.vstack((xs_multi[-1, node_uncontrol], sH, sL)).transpose()

    des = '../data/' + dynamics + '/threshold/threshold_sH_sL_multi_original/' 
    des_file = des + network_type + f'_N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_controlnum={control_num}.csv'
    if not os.path.exists(des):
        os.makedirs(des)
    df = pd.DataFrame(data)
    df.to_csv(des_file, index=False, header=False)
    return None


 

def threshold_sH_or_sL_neighbor(network_type, N, weight, network_seed, d, dynamics, arguments, threshold):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
    beta, _ = np.round(betaspace(A, [0]), 5)
    xL, xH = odeint(mutual_1D, np.array([0.1, 5]), np.arange(0, 1000, 0.01), args=(beta, arguments))[-1]

    sH_list = [0, 10]
    sL_list = [0, 100]
    while np.diff(sH_list)[0] > 1e-5:
        xs_left = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, sH_list[0], 0, arguments))[-1]
        xs_right = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, sH_list[1], 0, arguments))[-1]
        sH_list[0], sH_list[1] = binary_search(sH_list[0], sH_list[1], xs_left, xs_right, threshold)
    while np.diff(sL_list)[0] > 1e-5:
        xs_left = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, 0, sL_list[0], arguments))[-1]
        xs_right = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, 0, sL_list[1], arguments))[-1]
        sL_list[0], sL_list[1] = binary_search(sL_list[0], sL_list[1], xs_left, xs_right, threshold)
    sL = sL_list[-1]
    sH = sH_list[-1]

    threshold_des = '../data/' + dynamics + '/threshold/' 
    file_threshold = threshold_des +  f'ER.csv'
    if not os.path.exists(threshold_des):
        os.makedirs(threshold_des)

    data = np.hstack((network_seed, N, d, weight, beta, xL, xH, sL, sH))
    data_df = pd.DataFrame(data.reshape(1, len(data)))
    data_df.to_csv(file_threshold, index=False, header=False, mode='a')
    return sH_list, sL_list

def threshold_sH_or_sL_beta(dynamics, beta):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    xL, xH = odeint(mutual_1D, np.array([0.1, 5]), np.arange(0, 1000, 0.01), args=(beta, arguments))[-1]

    sH_list = [0, 10]
    sL_list = [0, 100]
    while np.diff(sH_list)[0] > 1e-5:
        xs_left = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, sH_list[0], 0, arguments))[-1]
        xs_right = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, sH_list[1], 0, arguments))[-1]
        sH_list[0], sH_list[1] = binary_search(sH_list[0], sH_list[1], xs_left, xs_right, threshold)
    while np.diff(sL_list)[0] > 1e-5:
        xs_left = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, 0, sL_list[0], arguments))[-1]
        xs_right = odeint(dynamics_xH_xL, np.array([0.1]), np.arange(0, 500, 0.01), args=(beta, xH, xL, 0, sL_list[1], arguments))[-1]
        sL_list[0], sL_list[1] = binary_search(sL_list[0], sL_list[1], xs_left, xs_right, threshold)
    sL = sL_list[-1]
    sH = sH_list[-1]

    threshold_des = '../data/' + dynamics + '/threshold/' 
    file_threshold = threshold_des +  f'sL_sH_beta.csv'
    if not os.path.exists(threshold_des):
        os.makedirs(threshold_des)

    data = np.hstack((beta, xL, xH, sL, sH))
    data_df = pd.DataFrame(data.reshape(1, len(data)))
    data_df.to_csv(file_threshold, index=False, header=False, mode='a')
    return sL, sH 

def nH_k_sL_sH(dynamics, beta, weight):
    """TODO: Docstring for nH_k_sL_sH.

    :dynamics: TODO
    :beta: TODO
    :returns: TODO

    """
    sL_c, sH_c = threshold_sH_or_sL_beta(dynamics, beta)
    a = sH_c / sL_c
    nH_list = []
    k_list = np.arange(0, 500, 1)
    for k in k_list:
        nH_left = 0
        nH_right = k
        if (k * weight) < sH_c:
            nH_c = -1
        elif k * weight * a > sH_c:
            nH_c = 0
        else:
            while int(nH_left) < int(nH_right):
                s_left = (int(nH_left) + (k-int(nH_left)) * a) * weight
                s_right = (int(nH_right) + (k-int(nH_right))  * a) * weight
                if s_right == sH_c:
                    break
                elif s_left == sH_c:
                    nH_right = int(np.ceil(nH_left))
                    break
                elif s_right > sH_c and s_left < sH_c :
                    nH_right = (nH_left + nH_right) / 2
                elif s_right < sH_c:
                    nH_left, nH_right = nH_right, nH_right*2 - nH_left
                elif s_left > sH_c:
                    nH_left, nH_right = nH_left * 2 - nH_right, nH_left
            nH_c = np.ceil(nH_right)
        nH_list.append(int(nH_c))
    threshold_nH_file = '../data/' + dynamics +  f'/threshold/beta={beta}_w={weight}.csv'
    data = np.vstack((k_list, nH_list))
    threshold_df = pd.DataFrame(data.transpose())
    threshold_df.to_csv(threshold_nH_file, index=False, header=False)

    return nH_list
        


def ER_degree(c, k):
    """TODO: Docstring for degree_distribution.

    :c: TODO
    :k: TODO
    :returns: TODO

    """
    return np.exp(-c) * c**k /math.factorial(k)

def empirical_degree(network_type, N, network_seed, d):
    """TODO: Docstring for empirical_degree.

    :network_type: TODO
    :N: TODO
    :network_seed: TODO
    :d: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, network_seed, d)
    degree_list = np.sum(A>0, 0)
    frequency, degree_bins = np.histogram(degree_list, bins=np.arange(degree_list.min(), degree_list.max()+2, 1))
    degree_probability = frequency/np.sum(frequency)
    degree_excess = (degree_bins[:-1] ) * degree_probability
    degree_excess_probability = degree_excess / np.sum(degree_excess)
    return degree_bins[:-1], degree_probability, np.hstack((degree_excess_probability[1:], 0))

def active_Z(x, degree_excess_distribution, f, n):
    """TODO: Docstring for active_probability.

    :f: TODO
    :returns: TODO

    """

    N = 500
    p = 0
    for k in range(0, N):
        if degree_excess_distribution[k]:
            p_comb = 1-np.sum([comb(k, l) *x**l * (1-x)**(k-l) for l in range(0, n[k])]) if k>=n[k] and n[k] >-0.1 else 0 
            p += degree_excess_distribution[k] * p_comb
    result = f + (1-f) * p - x
    return result

def active_R(Z, degree_distribution, f, n):
    """TODO: Docstring for active_probability.

    :f: TODO
    :returns: TODO

    """

    N = 500
    p = 0
    for k in range(0, N):
        if degree_distribution[k]:
            p_comb = 1-np.sum([comb(k, l) *Z**l * (1-Z)**(k-l) for l in range(0, n[k])]) if k>=n[k] and n[k] >-0.1 else 0 
            p += degree_distribution[k] * p_comb
    R = f + (1-f) * p 
    return R

def active_probability(degree_distribution, degree_excess_distribution, percolation_file, x_try, f, n):
    """TODO: Docstring for active_probability.

    :arg1: TODO
    :returns: TODO

    """
    x_solution = []
    for x0 in x_try:
        x_sol = fsolve(active_Z, x0, args=(degree_excess_distribution, f, n))
        result = active_Z(x_sol, degree_excess_distribution, f, n)
        if abs(result) < 1e-10:
            x_solution.append(x_sol)
    x_solution = np.array(x_solution)
    solution = np.min(x_solution[x_solution>=0])
    R = active_R(solution, degree_distribution, f, n)
    active_data = pd.DataFrame(np.hstack((f, R)).reshape(1, 2))
    active_data.to_csv(percolation_file, mode='a', index=False, header=False)
    return None

def active_transition(network_type, N, d, network_seed, beta, betaeffect, dynamics, x_try, f_list, degree_data):
    """TODO: Docstring for active_transition.

    :x_try: TODO
    :f_list: TODO
    :c: TODO
    :k: TODO
    :returns: TODO

    """
    k = 500
    degree = range(k)
    if degree_data=='empirical':
        degree_bins, degree_probability, degree_excess_probability = empirical_degree(network_type, N, network_seed, d)
        degree_distribution = np.hstack((np.zeros(degree_bins[0]), degree_probability, np.zeros(k-1-degree_bins[-1])))
        degree_excess_distribution = np.hstack((np.zeros(degree_bins[0] ), degree_excess_probability, np.zeros(k-1-degree_bins[-1])))
    elif network_type == 'ER':
        degree_distribution = np.array([ER_degree(c, i) for i in degree])

    percolation_dir = '../data/' + dynamics + '/' + network_type +  f'/percolation/'
    if not os.path.exists(percolation_dir):
        os.makedirs(percolation_dir)
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    if betaeffect:
        weight = A.max()
        percolation_file = percolation_dir + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_percolation.csv'
    else:
        weight = beta
        beta = betaspace(A, [0])[0]
        percolation_file = percolation_dir + f'N={N}_d={d}_netseed={network_seed}_weight={weight}_percolation.csv'
    n_H = nH_k_sL_sH(dynamics, beta, weight)
    p = mp.Pool(cpu_number)
    p.starmap_async(active_probability, [(degree_distribution, degree_excess_distribution, percolation_file, x_try, f, n_H) for f in f_list]).get()
    p.close()
    p.join()

    return None

def activate_process(active_seed, active_file, nodes, all_neighbors, f, n):
    """TODO: Docstring for activate_process.
    :returns: TODO

    """
    active_file = active_file + f'controlseed={active_seed}.csv'
    N_actual = len(nodes)
    random_state = np.random.RandomState(active_seed)
    control_order = random_state.choice(N_actual, N_actual, replace=False)
    node_r = control_order[:int(f*N_actual)]
    node_uncontrol = np.setdiff1d(control_order, node_r)

    state = np.zeros(N_actual)
    state[node_r] = 1
    state_a = 0
    state_b = N_actual
    while state_a != state_b:
        node_inactive = nodes[np.where(state ==0)[0]]
        state_a = np.sum(state)
        for node in node_inactive:
            neighbor = all_neighbors[node]
            degree = len(neighbor)
            if sum(state[neighbor]) >= n[degree] and n[degree] >= -0.1:
                state[node] = 1
        state_b = np.sum(state)
    data = np.hstack((f, state))
    active_data = pd.DataFrame(data.reshape(1, len(data)))
    active_data.to_csv(active_file , mode='a', index=False, header=False)
    return None

def activate_parallel(network_type, N, d, network_seed, control_seed_list, f, beta, betaeffect):
    """TODO: Docstring for activate_parallel.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    active_des = '../data/' + dynamics + '/' + network_type +  f'/percolation_activation/'
    if not os.path.exists(active_des):
        os.makedirs(active_des)
    if betaeffect == 0:
        weight = beta 
        beta = betaspace(A, [0])[0]
        active_file = active_des + f'N={N}_d={d}_netseed={network_seed}_weight={weight}_'
    else:
        weight = A.max()
        active_file = active_des + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_'

    G = nx.from_numpy_matrix(A)
    nodes = np.array(list(G.nodes()))
    all_neighbors = {}
    for node in nodes:
        all_neighbors[node] = list(G.neighbors(node))

    #nH = np.array(pd.read_csv('../data/' +dynamics + f'/threshold/beta={beta}_w={weight}.csv', header=None), dtype=int)[:, -1]
    nH = nH_k_sL_sH(dynamics, beta, weight)

    p = mp.Pool(cpu_number)
    p.starmap_async(activate_process, [(control_seed, active_file, nodes, all_neighbors, f, nH) for control_seed in control_seed_list]).get()
    p.close()
    p.join()
    return None

def heatmap_activate(network_type, N, d, network_seed, f, weight, active_seed, step_num):
    """TODO: Docstring for activate_process.
    :returns: TODO

    """

    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
    beta = np.round(betaspace(A, [0])[0], 5)
    n = np.array(pd.read_csv('../data/' +dynamics + f'/threshold/beta={beta}_w={weight}.csv', header=None), dtype=int)[:, -1]
    G = nx.from_numpy_matrix(A)
    nodes = np.array(list(G.nodes()))
    all_neighbors = {}
    for node in nodes:
        all_neighbors[node] = list(G.neighbors(node))

    N_actual = len(nodes)
    random_state = np.random.RandomState(active_seed)
    control_order = random_state.choice(N_actual, N_actual, replace=False)
    node_r = control_order[:int(f*N_actual)]
    node_uncontrol = np.setdiff1d(control_order, node_r)

    state = np.zeros(N_actual)
    state[node_r] = 1
    state_a = 0
    state_b = N_actual
    state_after = state.copy()
    state_temp = []
    while state_a != state_b:
        node_inactive = nodes[np.where(state ==0)[0]]
        state_a = np.sum(state)
        for node in node_inactive:
            neighbor = all_neighbors[node]
            degree = len(neighbor)
            if sum(state[neighbor]) >= n[degree+1] and n[degree] >= -0.1:
                state_after[node] = 1
        state_temp.append(state.copy())
        state = state_after.copy()
        state_b = np.sum(state)
    data_snap = state_temp[step_num]
    cmap = plt.cm.coolwarm
    nx.draw(G, pos= nx.spring_layout(G, seed=0) , cmap=cmap, node_color=data_snap)

    return state_temp



 


arguments = (B, C, D, E, H, K)
dynamics = 'mutual'
network_type = 'ER'
network_seed = 0
N = 1000
d = 8000
egwt_list = np.array([0.05, 0.1, 0.2])
degree_list = np.arange(0, 100, 1)
#neighbor_H = threshold_neighbor(network_type, N, egwt_list, network_seed, d, dynamics, arguments, degree_list)

x_try = np.arange(0, 1.1, 0.1)
degree_data = 'empirical'
c = 16
d = int(N*c/2)
weight = 0.1
f_list = np.arange(0.01, 1, 0.01)
beta = 1
beta_list = [5, 10, 20]
for beta in beta_list:
    #threshold_sH_sL_neighbor(dynamics, arguments, beta)
    pass
#threshold_sH_sL_neighbor(network_type, N, weight, network_seed, d, dynamics, arguments)
weight = 0.05
beta = 0.84706
threshold = 5
#threshold_sH_or_sL_neighbor(network_type, N, weight, network_seed, d, dynamics, arguments, threshold)
beta = 1
weight = 0.02237
#threshold_sH_or_sL_beta(dynamics, beta)
#nH_list = nH_k_sL_sH(dynamics, beta, weight)

f_list = np.round(np.arange(0.01, 1, 0.01), 2)
control_seed_list = [0, 1, 2, 3]
d = 4000
weight = 0.1
control_seed = 0
active_seed = 0
f = 0.2
N = 1000
d = 4000
step_num = 8
#state_temp = heatmap_activate(network_type, N, d, network_seed, f, weight, active_seed, step_num)


control_seed = 2
control_num = 0.1
control_seed_list = np.arange(10).tolist()
control_num_list = [0.1, 0.2]
attractor_high = 5
attractor_low = 0.1

dynamics = 'mutual'
beta = 1
N = 1000






network_type = 'ER'
d = 3000
network_seed_list = np.arange(0, 10, 1).tolist()
network_seed = 0

network_type = 'SF'
d = [2.5, 999, 3]
network_seed_list = np.tile(np.arange(0, 10, 1), (2, 1)).transpose().tolist()
network_seed = [0, 0]

f_list = np.arange(0, 1, 0.01)

betaeffect = 0
beta = 0.1


active_transition(network_type, N, d, network_seed, beta, betaeffect, dynamics, x_try, f_list, degree_data)
for f in f_list:
    #activate_parallel(network_type, N, d, network_seed, control_seed_list, f, beta, betaeffect)
    pass

for control_num in f_list:
    #random_control_parallel(network_type, N, beta, betaeffect, network_seed, d, control_num, control_seed_list, attractor_high, attractor_low)
    pass

for control_num in control_num_list:
    for control_seed in control_seed_list:
        for network_seed in network_seed_list:
            #threshold_sH_sL_neighbor_multi(network_type, N, beta, network_seed, d, dynamics, arguments, attractor_low, attractor_high, control_seed, control_num)
            pass
