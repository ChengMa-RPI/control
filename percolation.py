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

def active_equation(x, degree_distribution, f, c, k):
    """TODO: Docstring for active_probability.

    :f: TODO
    :returns: TODO

    """

    #result = f + (1-f) * np.sum([degree_distribution(c, i) * np.sum([comb(i, l) *x**l * (1-x)**(i-l) for l in range(k, i) ]) for i in range(k, N) ]) - x
    N = 100
    p = 0
    for i in range(k, N):
        p += degree_distribution[i] * sum([comb(i, l) *x**l * (1-x)**(i-l) for l in range(k, i+1)])
    result = f + (1-f) * p - x
    return result

def active_probability(degree_distribution, active_dir, x_try, f, c, k):
    """TODO: Docstring for active_probability.

    :arg1: TODO
    :returns: TODO

    """
    x_solution = []
    for x0 in x_try:
        x_sol = fsolve(active_equation, x0, args=(degree_distribution, f, c, k))
        result = active_equation(x_sol, degree_distribution, f, c ,k)
        if abs(result) < 1e-5:
            x_solution.append(x_sol)
    solution = np.min(x_solution)

    active_data = pd.DataFrame(np.hstack((f, solution)).reshape(1, 2))
    active_data.to_csv(active_dir + 'theory.csv', mode='a', index=False, header=False)
    return None

def active_transition(network_type, x_try, f_list, c, k):
    """TODO: Docstring for active_transition.

    :x_try: TODO
    :f_list: TODO
    :c: TODO
    :k: TODO
    :returns: TODO

    """
    N = 100
    degree = np.arange(N)
    if network_type == 'ER':
        degree_distribution = np.array([ER_degree(c, i) for i in degree])
    active_dir = '../data/percolation/' + network_type +  f'/c={c}/k={k}/'
    if not os.path.exists(active_dir):
        os.makedirs(active_dir)

    p = mp.Pool(cpu_number)
    p.starmap_async(active_probability, [(degree_distribution, active_dir, x_try, f, c, k) for f in f_list]).get()
    p.close()
    p.join()

    return None

def activate_process(active_seed, active_des, nodes, all_neighbors, f, k):
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
            if sum(state[neighbor]) >= k:
                state[node] = 1
        state_b = np.sum(state)

        t2 = time.time()
    active_fraction = state_a/N
    active_data = pd.DataFrame(np.hstack((active_seed, active_fraction)).reshape(1, 2))
    active_data.to_csv(active_des + f'_f={f}.csv', mode='a', index=False, header=False)
    return None


def activate_parallel(network_type, N, c, network_seed, active_seed_list, f, k):
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

    active_dir = '../data/percolation/' + network_type +  f'/c={c}/k={k}/'
    if not os.path.exists(active_dir):
        os.makedirs(active_dir)
    active_des = active_dir + 'netseed={network_seed}_N={N}'

    p = mp.Pool(cpu_number)
    p.starmap_async(activate_process, [(active_seed, active_des, nodes, all_neighbors, f, k) for active_seed in active_seed_list]).get()
    p.close()
    p.join()
    return None




network_type = 'ER'

x_try = np.arange(0, 1, 0.1)
f = 0.042
f_list = np.arange(0.04, 0.05, 0.001)
c = 4
k = 2 
#solution = active_probability(x_try, f, c, k)
#solution = active_transition(x_try, f_list, c, k)
active_transition(network_type, x_try, f_list, c, k)

N = 10000
network_seed = 0
active_seed_list = np.arange(100)

#activate_parallel(network_type, N, c, network_seed, active_seed_list, f, k)
