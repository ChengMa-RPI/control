import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, mutual_1D, mutual_multi, network_generate, normalization_x, ode_Cheng, gif

import numpy as np 
import math
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

cpu_number = 10

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
    return degree_bins[:-1], degree_probability

def active_equation(x, degree_distribution, f, n):
    """TODO: Docstring for active_probability.

    :f: TODO
    :returns: TODO

    """

    N = 100
    p = 0
    for k in range(0, N):
        #p += degree_distribution[i] * sum([comb(i, l) *x**l * (1-x)**(i-l) for l in range(k, i+1)])
        p_comb = 1-np.sum([comb(k, l) *x**l * (1-x)**(k-l) for l in range(0, n[k])]) if k>=n[k] else 0 
        p += degree_distribution[k] * p_comb
    result = f + (1-f) * p - x
    return result

def active_probability(degree_distribution, active_dir, x_try, f, n):
    """TODO: Docstring for active_probability.

    :arg1: TODO
    :returns: TODO

    """
    x_solution = []
    for x0 in x_try:
        x_sol = fsolve(active_equation, x0, args=(degree_distribution, f, n))
        result = active_equation(x_sol, degree_distribution, f, n)
        if abs(result) < 1e-10:
            x_solution.append(x_sol)
    x_solution = np.array(x_solution)
    solution = np.min(x_solution[x_solution>=0])

    active_data = pd.DataFrame(np.hstack((f, solution)).reshape(1, 2))
    active_data.to_csv(active_dir + 'theory.csv', mode='a', index=False, header=False)
    return None

def active_transition(network_type, x_try, f_list, c, beta, betaeffect, degree_data, N, network_seed, d):
    """TODO: Docstring for active_transition.

    :x_try: TODO
    :f_list: TODO
    :c: TODO
    :k: TODO
    :returns: TODO

    """
    k = 100
    degree = range(k)
    if degree_data=='empirical':
        degree_bins, degree_probability = empirical_degree(network_type, N, network_seed, d)
        degree_distribution = np.hstack((np.zeros(degree_bins[0]), degree_probability, np.zeros(k-1-degree_bins[-1])))
    elif network_type == 'ER':
        degree_distribution = np.array([ER_degree(c, i) for i in degree])

    active_dir = '../data/percolation/' + network_type +  f'/c={c}/'
    if betaeffect:
        active_dir += f'beta={beta}/'
    else:
        active_dir += f'edgewt={beta}/'
    n = np.array(pd.read_csv(active_dir + 'threshold.csv', header=None), dtype=int)[:, -1]
    p = mp.Pool(cpu_number)
    p.starmap_async(active_probability, [(degree_distribution, active_dir, x_try, f, n) for f in f_list]).get()
    p.close()
    p.join()

    return None

def activate_process(active_seed, active_des, nodes, all_neighbors, f, n):
    """TODO: Docstring for activate_process.
    :returns: TODO

    """
    random_state = np.random.RandomState(active_seed)
    state = np.zeros(N)
    initialize = random_state.choice(N, int(f*N), replace=False)
    state[initialize] = 1
    #initialize = random_state.random(N)
    #state[initialize<=f] = 1

    #node_state = {k: v for k, v in zip(nodes, state)}
    state_a = 0
    state_b = N
    while state_a != state_b:
        t1 = time.time()
        node_inactive = nodes[np.where(state ==0)[0]]
        state_a = np.sum(state)
        for node in node_inactive:
            neighbor = all_neighbors[node]
            degree = len(neighbor)
            if sum(state[neighbor]) >= n[degree]:
                state[node] = 1
        state_b = np.sum(state)

        t2 = time.time()
    active_fraction = state_a/N
    active_data = pd.DataFrame(np.hstack((active_seed, active_fraction)).reshape(1, 2))
    active_data.to_csv(active_des + f'_f={f}.csv', mode='a', index=False, header=False)
    return None

def activate_parallel(network_type, N, c, network_seed, active_seed_list, f, beta, betaeffect):
    """TODO: Docstring for activate_parallel.

    :arg1: TODO
    :returns: TODO

    """
    m = c * N /2  # the number of edges given the average degree c
    if network_type == 'ER':
        G = nx.gnm_random_graph(N, m, network_seed)
    nodes = np.array(list(G.nodes()))
    #adj_list = nx.generate_adjlist(G)
    all_neighbors = {}
    for node in nodes:
        all_neighbors[node] = list(G.neighbors(node))

    active_dir = '../data/percolation/' + network_type +  f'/c={c}/'
    if betaeffect:
        active_dir += f'beta={beta}/'
    else:
        active_dir += f'edgewt={beta}/'

    active_des = active_dir + f'netseed={network_seed}_N={N}'
    n = np.array(pd.read_csv(active_dir + 'threshold.csv', header=None), dtype=int)[:, -1]


    p = mp.Pool(cpu_number)
    p.starmap_async(activate_process, [(active_seed, active_des, nodes, all_neighbors, f, n) for active_seed in active_seed_list]).get()
    p.close()
    p.join()
    return None




network_type = 'ER'

x_try = np.arange(0, 1.1, 0.1)
f = 0.042
f_list = np.round(np.arange(0.0002, 0.01, 0.0001), 4)
f_list = np.round(np.arange(0.055, 0.09, 0.005), 3)
f_list = np.round(np.arange(0.044, 0.046, 0.0001), 4)
f_list = np.round(np.arange(0.01, 1, 0.01), 2)
f_list = np.round(np.arange(0.15, 0.20, 0.001), 3)
f_list = np.round(np.arange(0.05, 0.07, 0.001), 3)

degree_data = 'empirical'
N = 10000
network_seed = 0
c = 16
d = int(N*c/2)
beta = 0.05
betaeffect = 0
#active_transition(network_type, x_try, f_list, c, beta, betaeffect, degree_data, N, network_seed, d)

N = 10000
network_seed = 0
active_seed_list = np.arange(100)

f_list = np.round(np.arange(0.1, 0.149, 0.01), 2)

for f in f_list:
    activate_parallel(network_type, N, c, network_seed, active_seed_list, f, beta, betaeffect)
