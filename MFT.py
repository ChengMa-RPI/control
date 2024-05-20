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


def MFT_constant(x, t, beta, arguments):
    """TODO: Docstring for PQR_constant.

    :arg1: TODO
    :returns: TODO

    """
    B, C, D, E, H, K = arguments
    x[np.where(x<0)] = 0  # Negative x is forbidden
    f = B + x * (1 - x/K) * ( x/C - 1)
 
    xp, xq, xr = x
    f[-1] = 0.0
    beta_p, beta_pq, beta_pr, beta_q, beta_qp = beta 
    gp = beta_p * xp / (D + E*xp + H*xp) + beta_pq * xq / (D + E*xp + H*xq) + beta_pr * xr/ (D + E*xp + H*xr)
    gq = beta_q * xq / (D + E*xq + H*xq) + beta_qp * xp / ( D + E*xq + H*xp)
    gr = 0.0

    dxdt = f + x * np.array([gp, gq, gr])

    return dxdt

def beta_PQR(A, x, control_seed, control_num):
    """TODO: Docstring for beta_PQR.

    :A: TODO
    :x: TODO
    :returns: TODO

    """
    N = np.size(A, 0)
    random_state = np.random.RandomState(control_seed)
    node_r = random_state.choice(N, int(control_num*N), replace=False)
    node_p = []
    for r in node_r:
        node_p.append(np.where(A[r]>0)[0])
    node_p = np.unique(np.hstack(node_p))
    node_p = np.setdiff1d(node_p, node_r)
    node_q = np.setdiff1d(np.arange(N), np.hstack((node_p, node_r)))

    node_pq = np.hstack((node_p, node_q))
    A_PQPQ = np.zeros((N, N))
    A_PQPQ[np.ix_(node_pq, node_pq)] = A[node_pq][:, node_pq]
    s_pq_out = np.sum(A_PQPQ, 1)

    "s_p_out"
    #s_p_out = np.sum(A, 1)[node_p]
    s_p_out = s_pq_out[node_p]
    A_P = A[node_p][:, node_p]
    s_p_in = np.sum(A_P, 0)
    A_RP = A[node_r][:, node_p]
    s_pr_in = np.sum(A_RP, 0)
    beta_p = np.sum(s_p_out * s_p_in)/np.sum(s_p_out)
    beta_pr = np.sum(s_p_out * s_pr_in)/np.sum(s_p_out)

    if len(node_q):
        s_q_out = s_pq_out[node_q]

        A_QP = A[node_q][:, node_p]
        s_pq_in = np.sum(A_QP, 0)

        A_Q = A[node_q][:, node_q]
        s_q_in = np.sum(A_Q, 0)

        A_PQ = A[node_p][:, node_q]
        s_qp_in = np.sum(A_PQ, 0)

        beta_pq = np.sum(s_p_out * s_pq_in)/np.sum(s_p_out)
        beta_q = np.sum(s_q_out * s_q_in)/np.sum(s_q_out)
        beta_qp = np.sum(s_q_out * s_qp_in)/np.sum(s_q_out)
    else:
        beta_pq = 0
        beta_q = 0
        beta_qp = 0


    beta = np.array([beta_p, beta_pq, beta_pr, beta_q, beta_qp])
    if len(x):
        x_p = x[node_p]
        x_P = np.sum(s_p_out * x_p ) / np.sum(s_p_out)
        if len(node_q):
            x_q = x[node_q]
            x_Q = np.sum(s_q_out * x_q ) / np.sum(s_q_out)
        else:
            x_Q = 0
        return beta, x_P, x_Q
    else:
        return beta

def MFT_dynamis(A, control_seed, control_num, control_value, t, arguments, MFT_des):
    """TODO: Docstring for MFT_dynamis.

    :arg1: TODO
    :returns: TODO

    """
    beta_eff = beta_PQR(A, [], control_seed, control_num)
    #x = ode_Cheng(MFT_PQR_constant, np.array([0.1, 0.1, control_value]), t, *(beta_eff, arguments))
    x = odeint(MFT_constant, np.array([0.1, 0.1, control_value]), t, args=(beta_eff, arguments))
    data = np.hstack((control_seed, x[-1], beta_eff))
    data_df = pd.DataFrame(data.reshape(1, len(data)))

    data_df.to_csv(MFT_des + '.csv', mode = 'a', index=False, header=False)

    return None

def MFT_parallel(dynamics, network_type, N, arguments, beta, betaeffect, control_num, control_value, control_seed_list, network_seed, d, t):
    """TODO: Docstring for MFT_parallel.

    :arg1: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)

    average_degree = int(2 * d/np.size(A, 0))
    MFT_dir = '../data/' +  dynamics.__name__ + '/' + network_type + f'/c={average_degree}/'
    if betaeffect:
        MFT_dir = MFT_dir + f'/beta={beta}/value={control_value}/'
    else:
        MFT_dir = MFT_dir + f'/edgewt={beta}/value={control_value}/'
    if not os.path.exists(MFT_dir):
        os.makedirs(MFT_dir)

    MFT_des = MFT_dir + f'netseed={network_seed}_N={N}_control_num={control_num}'

    p = mp.Pool(cpu_number)
    p.starmap_async(MFT_dynamis, [(A, control_seed, control_num, control_value, t, arguments, MFT_des) for control_seed in control_seed_list]).get()
    p.close()
    p.join()
    return None

def reduction_PQ(dynamics_multi, network_type, N, beta, betaeffect, network_seed, d):
    """TODO: Docstring for reduction_PQ.

    :network_type: TODO
    :N: TODO
    :beta: TODO
    :betaeffect: TODO
    :network_seed: TODO
    :d: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    c = int(d*2/N)
    multi_dir = '../data/' +  dynamics_multi + '/' + network_type + f'/c={c}/'
    if betaeffect:
        multi_dir = multi_dir + f'/beta={beta}/value={control_value}/'
    else:
        multi_dir = multi_dir + f'/edgewt={beta}/value={control_value}/'

    multi_des = multi_dir + f'netseed={network_seed}_N={N}_control_num={control_num}.csv'

    multi = np.array(pd.read_csv(multi_des, header=None).iloc[:, :])
    multi_seed = np.array(multi[:, 0], int)
    multi_sort = multi[:, 1:][np.argsort(multi_seed)]

    PQ_eff = []
    for control_seed in np.sort(multi_seed):
        beta_eff, x_P, x_Q = beta_PQR(A, multi_sort[control_seed], control_seed, control_num)
        PQ_eff.append([x_P, x_Q])
    PQ_eff = np.vstack((PQ_eff))
    data = np.vstack((np.sort(multi_seed), PQ_eff.transpose()))
    data_df = pd.DataFrame(data.transpose())

    reduction_dir = multi_dir + 'reduction/'
    if not os.path.exists(reduction_dir):
        os.makedirs(reduction_dir)
    
    data_df.to_csv(reduction_dir + f'netseed={network_seed}_N={N}_control_num={control_num}.csv', mode = 'a', index=False, header=False)

    return None

    
        


dynamics = MFT_constant
dynamics_multi = 'mutual_constant'
arguments = (B, C, D, E, H, K)
network_type = 'ER'
N = 1000
beta = 0.1
betaeffect = 0
network_seed = 0
c = 16
d = int(N*c/2)
control_seed_list = np.arange(100)
control_num_list = [0.99]
control_num_list = np.round(np.arange(0.01, 0.061, 0.01), 2)
control_value = 5
t = np.arange(0, 500, 0.01)


for control_num in control_num_list:

    #MFT_parallel(dynamics, network_type, N, arguments, beta, betaeffect, control_num, control_value, control_seed_list, network_seed, d, t)
    reduction_PQ(dynamics_multi, network_type, N, beta, betaeffect, network_seed, d)
