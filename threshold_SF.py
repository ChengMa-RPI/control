import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, network_generate, normalization_x, ode_Cheng, gif

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
import collections
from functools import reduce

B = 0.1
C = 1
K_mutual = 5
D = 5 
E = 0.9
H = 0.1

r= 1
K= 10
c = 2.4
B_gene = 1 
B_SIS = 1
B_BDP = 1
B_PPI = 1
F_PPI = 0.1
f = 1
h = 2
a = 5
b = 1


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


def mutual_multi(x, t, arguments, net_arguments):
    B, C, D, E, H, K = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = A_interaction * x[index_j] / (D + E * x[index_i] + H * x[index_j])
    dxdt = sum_f + x * np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def mutual_1D(x, t, c, arguments):
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + c * x**2 / (D + (E+H) * x)
    return dxdt

def mutual_decouple(x, t, x_eff, w, arguments):
    B, C, D, E, H, K = arguments
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = w * x * x_eff / (D + E * x + H * x_eff)
    dxdt = sum_f + sum_g
    return dxdt

def mutual_xH_xL(x, t, x_h, x_l, k_H, k_L, arguments):
    """TODO: mutualistic dynamics with k_H neighbors in x_h and k_L neighbors in x_l.

    """
    B, C, D, E, H, K = arguments
    fx = B + x * (1 - x/K) * ( x/C - 1) 
    gx = k_H*x*x_h / (D + E*x + H*x_h) + k_L*x*x_l/(D + E*x + H*x_l)
    dxdt = fx + gx
    return dxdt

def mutual_G(xi, xj, arguments):
    """TODO: interaction for mutualistic dynamics.

    """
    B, C, D, E, H, K = arguments
    gx = xi*xj / (D + E*xi + H*xj)
    return gx


def genereg_multi(x, t, arguments, net_arguments):
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - B * x 
    sum_g = A_interaction * x[index_j]**2/(x[index_j]**2+1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def genereg_1D(x, t, c, arguments):
    B, = arguments
    dxdt = -B * x  + c * x**2 / (x**2 + 1)
    return dxdt

def genereg_decouple(x, t, x_eff, w, arguments):
    B, = arguments
    sum_f = - B * x 
    sum_g = w * x_eff**2/(x_eff**2+1)
    dxdt = sum_f + sum_g
    return dxdt

def genereg_xH_xL(x, t, x_h, x_l, k_H, k_L, arguments):
    """TODO: gene regulatory dynamics with k_H neighbors in x_h and k_L neighbors in x_l.

    """
    B, = arguments
    fx = - B * x 
    gx = k_H * x_h**2 / (x_h**2+1) + k_L * x_l**2 / (x_l**2+1)
    dxdt = fx + gx
    return dxdt

def genereg_G(xi, xj, arguments):
    """TODO: interaction for mutualistic dynamics.

    """
    B, = arguments
    gx = xj**2 / (xj**2+1)
    return gx

def genereg_multi_constant(x, t, control_node, control_constant, arguments, net_arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[control_node] = control_constant 
    sum_f = -B*x
    sum_g = A_interaction * x[index_j]**2/ (x[index_j]**2 + 1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    dxdt[control_node] = 0.0
    return dxdt

def harvest_multi(x, t, arguments, net_arguments):
    r, K, c = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    sum_g = A_interaction * (x[index_j] - x[index_i])
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt

def harvest_1D(x, t, beta_c, arguments):
    r, K, c = arguments
    dxdt = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    return dxdt

def harvest_decouple(x, t, x_eff, w, arguments):
    r, K, c = arguments
    sum_f = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    sum_g = w * (x_eff - x)
    dxdt = sum_f + sum_g
    return dxdt

def harvest_xH_xL(x, t, x_h, x_l, k_H, k_L, arguments):
    r, K, c = arguments
    fx = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    gx = k_H * (x_h - x) + k_L * (x_l - x)
    dxdt = fx + gx
    return dxdt

def harvest_G(xi, xj, arguments):
    r, K, c = arguments
    gx = xj - xi
    return gx


def harvest_multi_constant(x, t, control_node, control_constant, arguments, net_arguments):
    r, K, c = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    x[control_node] = control_constant 
    sum_f = r * x * (1 - x/K) - c * x**2 / (x**2 + 1)
    sum_g = A_interaction * (x[index_j] - x[index_i])
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    dxdt[control_node] = 0.0
    return dxdt


def threshold_sH_or_sL_diffstate(network_type, N, d, beta, betaeffect, network_seed, dynamics, arguments, attractor_high, attractor_low):
    """original dynamics N species interaction.

    :returns: derivative of x 

    """
    dynamics_decouple = globals()[dynamics + '_decouple']
    dynamics_1D = globals()[dynamics + '_1D']
    dynamics_G = globals()[dynamics + '_G']
    dynamics_xH_xL = globals()[dynamics + '_xH_xL']
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    beta_calculate = betaspace(A, [0])[0]
    w = np.sum(A, 0)
    N_actual = len(A)
    initial_low = np.ones(N_actual) * attractor_low
    initial_high = np.ones(N_actual) * attractor_high
    t = np.arange(0, 1000, 0.01)
    xL, xH = odeint(dynamics_1D, np.array([attractor_low, attractor_high]), np.arange(0, 1000, 0.01), args=(beta_calculate, arguments))[-1]
    xL_decouple = odeint(dynamics_decouple, initial_low, t, args=(xL, w, arguments))[-1]
    xH_decouple = odeint(dynamics_decouple, initial_high, t, args=(xH, w, arguments))[-1]

    sH_interval = [0, 100]
    while np.diff(sH_interval)[0] > 1e-5:
        xs_left = odeint(dynamics_xH_xL, np.array([attractor_low]), np.arange(0, 500, 0.01), args=(xH, xL, sH_interval[0], 0, arguments))[-1]
        xs_right = odeint(dynamics_xH_xL, np.array([attractor_low]), np.arange(0, 500, 0.01), args=(xH, xL, sH_interval[1], 0, arguments))[-1]
        sH_interval[0], sH_interval[1] = binary_search(sH_interval[0], sH_interval[1], xs_left, xs_right, attractor_high)
    sH = sH_interval[-1]
    xs_critical = xs_right
    g_contribute = dynamics_G(xs_critical, xH_decouple, arguments)
    gH_standard  = dynamics_G(xs_critical, xH, arguments)
    gL_standard  = dynamics_G(xs_critical, xL, arguments)
    gi_gH_ratio = g_contribute / gH_standard
    gL_gH_ratio = gL_standard / gH_standard
        
    return sH, gL_gH_ratio, gi_gH_ratio


def multi_conv(arr, iter_times):
    if iter_times == 0:
        temp_result = np.array([1])
    else:
        temp_result = arr
        for _ in range(1, iter_times):
            temp_result = np.convolve(arr, temp_result, mode='full') 
    return temp_result

def threshold_probability(network_type, N, d, network_seed, beta, betaeffect, dynamics, arguments, bin_num=100):
    """TODO: Docstring for threshold_probability.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :network_seed: TODO
    :dynamics: TODO
    :arguments: TODO
    :num_high_neighbor: TODO
    :n_threshold: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    weight = A.max()  # for uniform edge weights
    k = np.sum(A>0, 0)
    N_actual = len(k)
    k_unique, index_unique = np.unique(k, return_index=True)
    k_count = dict(sorted(collections.Counter(k).items()))
    excess_degree = np.array(list(k_count.keys())) * np.array(list(k_count.values()))
    excess_degree_dis = {i:j for i, j in zip(k_count.keys(), excess_degree)}

    sH, gL_gH_ratio, gi_gH_ratio = threshold_sH_or_sL_diffstate(network_type, N, d, beta, betaeffect, network_seed, dynamics, arguments, attractor_high, attractor_low)
    n_threshold = sH / weight
    gi_gH_ratio_unique = gi_gH_ratio[index_unique]
    n_dist, n_interval = np.histogram(gi_gH_ratio_unique, bin_num, weights=excess_degree, density=True)
    n_value = (n_interval[:-1] + n_interval[1:]) / 2
    n_value_diff = np.mean(np.diff(n_value))
    survive_prob = {}
    for num_high_neighbor in range(k_unique.max()+1):
        if not num_high_neighbor in survive_prob:
            survive_prob[num_high_neighbor] = {}
        n_high_sum_value = np.array([n_value[0]*num_high_neighbor + i * n_value_diff for i in range(num_high_neighbor * (bin_num-1)+ 1)])
        n_high_sum_dis = multi_conv(n_dist, num_high_neighbor)
        n_high_sum_prob = n_high_sum_dis/n_high_sum_dis.sum()
        for k_i in k_unique:
            num_low_neighbor = k_i - num_high_neighbor
            if num_low_neighbor >= 0:
                contribute_low_neighbor = num_low_neighbor * gL_gH_ratio
                if n_high_sum_value.max() < (n_threshold-contribute_low_neighbor):
                    n_high_sum_cumu_prob = 0
                else:
                    index_satisfy = np.where(n_high_sum_value >= (n_threshold-contribute_low_neighbor))[0][0]
                    n_high_sum_cumu_prob = np.sum(n_high_sum_prob[index_satisfy:])
                survive_prob[num_high_neighbor][k_i] = n_high_sum_cumu_prob
        if abs(min(list(survive_prob[num_high_neighbor].values())) - 1) < 1e-8:
            break
    return survive_prob

def threshold_probability_costomize(network_type, N, d, network_seed, beta, betaeffect, dynamics, arguments, bin_num=100):
    """TODO: Docstring for threshold_probability.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :network_seed: TODO
    :dynamics: TODO
    :arguments: TODO
    :num_high_neighbor: TODO
    :n_threshold: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    weight = A.max()  # for uniform edge weights
    k = np.sum(A>0, 0)
    N_actual = len(k)
    k_unique, index_unique = np.unique(k, return_index=True)
    k_count = dict(sorted(collections.Counter(k).items()))

    sH, gL_gH_ratio, gi_gH_ratio = threshold_sH_or_sL_diffstate(network_type, N, d, beta, betaeffect, network_seed, dynamics, arguments, attractor_high, attractor_low)
    n_threshold = sH / weight
    gi_gH_ratio_unique = gi_gH_ratio[index_unique]
    survive_prob = {}
    for i in range(N_actual):
        neighbor_index = np.where(A[i] > 0)[0]
        neighbor_k_i = k[neighbor_index]
        weights = []
        for j in k_unique:
            weights.append(sum(neighbor_k_i == j))
        n_dist, n_interval = np.histogram(gi_gH_ratio_unique, bin_num, weights=weights, density=True)
        n_value = (n_interval[:-1] + n_interval[1:]) / 2
        n_value_diff = np.mean(np.diff(n_value))
        for num_high_neighbor in range(len(neighbor_index)+1):
            if not num_high_neighbor in survive_prob:
                survive_prob[num_high_neighbor] = {}
            n_high_sum_value = np.array([n_value[0]*num_high_neighbor + i * n_value_diff for i in range(num_high_neighbor * (bin_num-1)+ 1)])
            n_high_sum_dis = multi_conv(n_dist, num_high_neighbor)
            n_high_sum_prob = n_high_sum_dis/n_high_sum_dis.sum()
            num_low_neighbor = len(neighbor_index) - num_high_neighbor
            contribute_low_neighbor = num_low_neighbor * gL_gH_ratio
            if n_high_sum_value.max() < (n_threshold-contribute_low_neighbor):
                n_high_sum_cumu_prob = 0
            else:
                index_satisfy = np.where(n_high_sum_value >= (n_threshold-contribute_low_neighbor))[0][0]
                n_high_sum_cumu_prob = np.sum(n_high_sum_prob[index_satisfy:])
            survive_prob[num_high_neighbor][i] = n_high_sum_cumu_prob
            if abs(n_high_sum_cumu_prob- 1) < 1e-8:
                break
    survive_prob_ave_over_deg = collections.defaultdict(dict)
    for i, j in zip(survive_prob.keys(), survive_prob.values()):
        
        index_j, prob = np.array(list(j.keys())), np.array(list(j.values()))
        for k_j_unique in k_unique:
            index = np.where(k[index_j] == k_j_unique)[0]
            if len(index):
                survive_prob_ave_over_deg[i][k_j_unique] = np.mean(prob[index])
            else:
                survive_prob_ave_over_deg[i][k_j_unique] = 1
    return survive_prob_ave_over_deg

def activate_process_diffthreshold(active_seed, active_file, nodes, all_neighbors, f, gL_gH_ratio, gi_gH_ratio, nH_critical):
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
            neighbor_control = np.intersect1d(node_r, neighbor)
            neighbor_uncontrol = np.setdiff1d(neighbor, neighbor_control)
            neighbor_active_uncontrol = np.where(state[neighbor_uncontrol] == 1)[0]
            neighbor_active_contribute = gi_gH_ratio[neighbor[neighbor_active_uncontrol]] if len(neighbor_active_uncontrol) else [0]
            neighbor_inactive_num =  len(neighbor) - len(neighbor_control) - len(neighbor_active_uncontrol)
            nH_sum = neighbor_inactive_num * gL_gH_ratio + np.sum(neighbor_active_contribute) + len(neighbor_control)
            if nH_sum >= (nH_critical[node]):
                state[node] = 1
        state_b = np.sum(state)
    data = np.hstack((f, state))
    active_data = pd.DataFrame(data.reshape(1, len(data)))
    #active_data.to_csv(active_file , mode='a', index=False, header=False)
    return data




dynamics = 'mutual'
arguments = (B, C, D, E, H, K_mutual)
attractor_high = 5
attractor_low = 0.1

dynamics = 'harvest'
arguments = (r, K, c)
attractor_high = 5
attractor_low = 0.1

dynamics = 'genereg'
arguments = (B_gene, )
attractor_high = 10
attractor_low = 0.001

network_type = 'SF'
N = 1000
beta = 0.2
betaeffect = 0
d = [2.5, 0, 3]
network_seed_list = np.tile(np.arange(1), (2, 1)).transpose().tolist()
network_seed = [0, 0]

network_type = 'ER'
beta = 0.4
betaeffect = 0 
d = 3000
network_seed_list = [0]



k_mean  = []
for network_seed in network_seed_list:
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    beta_calculate = betaspace(A, [0])[0]
    k = np.sum(A>0, 0)
    N_actual = len(k)
    print(k.shape, sum(k==1))
    k_mean.append(np.mean(k))

G = nx.from_numpy_matrix(A)
nodes = np.array(list(G.nodes()))
all_neighbors = {}
for node in nodes:
    all_neighbors[node] = np.array(list(G.neighbors(node)))

ratio_threshold = 0.13
dynamics_decouple = globals()[dynamics + '_decouple']
dynamics_xH_xL = globals()[dynamics + '_xH_xL']
dynamics_1D = globals()[dynamics + '_1D']
dynamics_G = globals()[dynamics + '_G']
dynamics_multi_constant = globals()[dynamics + '_multi_constant']

net_arguments = (index_i, index_j, A_interaction, cum_index)
A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
beta_calculate = betaspace(A, [0])[0]
w = np.sum(A, 0)
N_actual = len(A)
initial_low = np.ones(N_actual) * attractor_low
initial_high = np.ones(N_actual) * attractor_high
t = np.arange(0, 1000, 0.01)
xL, xH = odeint(dynamics_1D, np.array([attractor_low, attractor_high]), np.arange(0, 1000, 0.01), args=(beta_calculate, arguments))[-1]
xL_decouple = odeint(dynamics_decouple, initial_low, t, args=(xL, w, arguments))[-1]
xH_decouple = odeint(dynamics_decouple, initial_high, t, args=(xH, w, arguments))[-1]

xs_critical = ratio_threshold * xH_decouple + (1 - ratio_threshold) * xL_decouple

"""
sH_list = []
for xs_critical_i in xs_critical:
    print(xs_critical_i)
    sH_interval = [0, 100]
    while np.diff(sH_interval)[0] > 1e-3:
        xs_left = odeint(dynamics_xH_xL, np.array([attractor_low]), np.arange(0, 500, 0.01), args=(xH, xL, sH_interval[0], 0, arguments))[-1]
        xs_right = odeint(dynamics_xH_xL, np.array([attractor_low]), np.arange(0, 500, 0.01), args=(xH, xL, sH_interval[1], 0, arguments))[-1]
        sH_interval[0], sH_interval[1] = binary_search(sH_interval[0], sH_interval[1], xs_left, xs_right, xs_critical_i)
    sH = sH_interval[-1]
    sH_list.append(sH)

nH_critical = np.array(sH_list) / beta
g_contribute = dynamics_G(xs_critical_i, xH_decouple, arguments)
gH_standard = dynamics_G(xs_critical_i, xH, arguments)
gL_standard = dynamics_G(xs_critical_i, xL, arguments)
gi_gH_ratio = g_contribute / gH_standard
gL_gH_ratio = gL_standard / gH_standard
     
"""
active_seed = 0
f = 0.01
active_file = ''
#active_data = activate_process_diffthreshold(active_seed, active_file, nodes, all_neighbors, f, gL_gH_ratio, gi_gH_ratio, nH_critical)
control_node = np.random.RandomState(active_seed).choice(N_actual, N_actual, replace=False)[:int(f*N_actual)]
control_constant = xH
xs_multi_control = odeint(dynamics_multi_constant, initial_low, np.arange(0, 500, 0.01), args=(control_node, control_constant, arguments, net_arguments))[-1]


#survive_prob = threshold_probability(network_type, N, d, network_seed, beta, betaeffect, dynamics, arguments)
#survive_prob_costomize = threshold_probability_costomize(network_type, N, d, network_seed, beta, betaeffect, dynamics, arguments, bin_num=100)

"""
neighbor_k_i = collections.defaultdict(list)
for network_seed in network_seed_list:
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    weight = A.max()  # for uniform edge weights
    k = np.sum(A>0, 0)
    N_actual = len(k)
    k_unique, index_unique = np.unique(k, return_index=True)
    k_count = dict(sorted(collections.Counter(k).items()))

    for k_i in k_unique:
        index_k_i = np.where(k == k_i)[0]
        for i in index_k_i:
            neighbor_k_i[k_i].extend(k[np.where(A[i]>0)[0]].tolist())
        #neighbor_k_i[k_i] = k[np.where(A[index_k_i]>0)[-1]]

neighbor_k_dis = {}
for i, j in zip(neighbor_k_i.keys(), neighbor_k_i.values()):
    neighbor_k_dis[i] = collections.Counter(j)

neighbor_k_dis = dict(sorted(neighbor_k_dis.items()))
for i in neighbor_k_dis.keys():
    if i<10:
        k_frequency = neighbor_k_dis[i]
        plt.loglog(list(k_frequency.keys()), list(k_frequency.values()), '.', label=i)
plt.legend()
#plt.show()

"""
