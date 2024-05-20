import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, mutual_1D, mutual_multi, network_generate, normalization_x, gif

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
import pathlib 
import itertools
from cycler import cycler
import matplotlib as mpl


fontsize = 22
fs = 22
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


mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-', '--']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))


def heatmap(dynamics, network_type, d, network_seed, beta, betaeffect, N, control_num, control_value, control_seed, plot_interval, dt=0.01, linewidth=0):
    """plot and save figure for animation

    :des: the destination where data is saved and the figures are put
    :realization_index: which data is chosen
    :plot_range: the last moment to plot 
    :plot_interval: the interval to plot
    :dt: simulation interval
    :returns: None

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)

    if network_type == '2D':
        evolution_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/beta={beta}/evolution/'
        evolution_des = evolution_dir + f'N={N}_control_num={control_num}_value={control_value}_control_seed={control_seed}.npy'
        heatmap_des = evolution_dir + f'N={N}_control_num={control_num}_value={control_value}/control_seed={control_seed}/'
    else:
        if betaeffect:
            evolution_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/d={d}/beta={beta}/evolution/'
        else:
            evolution_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/d={d}/edgewt={beta}/evolution/'
        evolution_des = evolution_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}_control_seed={control_seed}.npy'
        heatmap_des = evolution_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}/control_seed={control_seed}/'
        G = nx.from_numpy_matrix(A)
        position = nx.spring_layout(G )

    path = pathlib.Path(evolution_des)
    with path.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        data = np.load(f)
        while f.tell() < fsz:
            data = np.vstack((data, np.load(f)))
    if not os.path.exists(heatmap_des):
        os.makedirs(heatmap_des)
    rho = data
    for i in np.arange(0, int(np.size(data, 0) * dt), plot_interval):
        if network_type == '2D':
            data_snap = rho[int(i/dt)].reshape(int(np.sqrt(N)), int(np.sqrt(N)))
            fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth)
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=0.8 * fontsize)
            fig = fig.get_figure()

        else:
            data_snap = rho[int(i/dt)]
            fig = plt.figure()
            cmap = plt.cm.coolwarm
            vmin = 0
            vmax = 1

            nx.draw(G, pos=position, cmap=cmap, node_color=data_snap, vmin=vmin, vmax=vmax)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, shrink=.8)

        #fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth, cbar_kws = {"orientation" : "horizontal"})

        """
        data_snap = abs(data_snap)
        data_snap = np.log(data_snap)
        fig = sns.heatmap(data_snap, vmin=-4, vmax=0, linewidths=linewidth)
        """
        # plt.subplots_adjust(left=0.02, right=0.98, wspace=0.25, hspace=0.25, bottom=0.02, top=0.98)
        plt.subplots_adjust(left=0.15, right=0.88, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
        plt.axis('off')
        #fig.patch.set_alpha(0.)
        # plt.title('time = ' + str(round(i, 2)) )
        fig.savefig(heatmap_des + str(int(i/plot_interval)) + '.png', format="png")

        plt.close()
    return None

def plot_ratio_distribution(dynamics, network_type, d, k, network_seed, beta, betaeffect, N, control_num, control_value):
    """TODO: Docstring for plot_ratio_distribution.

    :dynamics: TODO
    :network_type: TODO
    :beta: TODO
    :N: TODO
    :control_num: TODO
    :control_value: TODO
    :: TODO
    :returns: TODO

    """


    if dynamics:
        ratio_dir = '../data/' +  dynamics.__name__ + '/' + network_type + f'/c={d}/'
        if betaeffect:
            ratio_dir = ratio_dir + f'beta={beta}/'
        else:
            ratio_dir = ratio_dir + f'edgewt={beta}/'

        ratio_des = ratio_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'

    else:
        ratio_dir = '../data/' +  'percolation'+ '/' + network_type + f'/c={d}/k={k}/'
        ratio_des = ratio_dir + f'netseed={network_seed}_N={N}_f={control_num}.csv'

    ratio = np.array(list(pd.read_csv(ratio_des, header=None).iloc[:, 1]))
    bins = np.arange(0, 1.01, 0.01)

    frequency, bins, patches = plt.hist(ratio, bins=bins, density=True, color='tab:red')
    plt.grid(axis='y', alpha=0.75)
    if dynamics:
        plt.xlabel('Transition ratio $R$', fontsize=fontsize)
    else:
        plt.xlabel('Activating probability $R$', fontsize=fontsize)

    plt.ylabel('Frequency %', fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    #plt.show()
    return ratio

def beta_influence(dynamics, network_type, N_list, beta_list, control_num_list, control_value, plot_type):
    """TODO: Docstring for beta_influence.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :beta_set: TODO
    :control_num: TODO
    :control_value: TODO
    :returns: TODO

    """
    ratio_des = '../data/' +  dynamics.__name__ + '_' + network_type 
    if np.size(control_num_list) > 1:
        ave_list = np.zeros((np.size(beta_list), np.size(control_num_list)))
        complete_list = np.zeros((np.size(beta_list), np.size(control_num_list)))
        N = N_list[0]
        labels = [f'M={control_num}' for control_num in control_num_list]
        for beta, i in zip(beta_list, range(np.size(beta_list))):
            for control_num, j in zip(control_num_list, range(np.size(control_num_list))):
                ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
                ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:1000, 1]))
                ratio_mean = np.mean(ratio)
                ratio_complete = np.sum(ratio == 1)/np.size(ratio)
                ave_list[i, j] = ratio_mean
                complete_list[i, j] = ratio_complete

    elif np.size(N_list) > 1:
        ave_list = np.zeros((np.size(N_list),  np.size(beta_list)))
        complete_list = np.zeros((np.size(N_list),  np.size(beta_list)))
        control_num = control_num_list[0]
    else:
        ave_list = np.zeros((np.size(beta_list)))
        complete_list = np.zeros((np.size(beta_list)))
        N = N_list[0]
        control_num = control_num_list[0]
        for beta, i in zip(beta_list, range(np.size(beta_list))):
            ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
            ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:1000, 1]))
            ratio_mean = np.mean(ratio)
            ratio_complete = np.sum(ratio == 1)/np.size(ratio)
            ave_list[i] = ratio_mean
            complete_list[i] = ratio_complete
    if plot_type == 'mean':
        feature = ave_list
        ylabel = 'average ratio'
    elif plot_type == 'complete':
        feature = complete_list
        ylabel = 'complete ratio'
    plt.plot(beta_list, feature, linewidth=lw, alpha = alpha)
    plt.xlabel('$\\beta$', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(labels, frameon=False, fontsize = legendsize)
    plt.show()

def control_num_influence(dynamics, network_type, d, network_seed, N_list, beta_list, control_num_list, control_value, plot_type):
    """TODO: Docstring for beta_influence.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :beta_set: TODO
    :control_num: TODO
    :control_value: TODO
    :returns: TODO

    """
    ratio_des = '../data/' +  dynamics.__name__ + '/' + network_type  + '/c={d}/'
    if np.size(beta_list) > 1:
        ave_list = np.zeros((np.size(control_num_list), np.size(beta_list)))
        complete_list = np.zeros((np.size(control_num_list), np.size(beta_list)))
        N = N_list[0]
        labels = [f'$\\beta={beta}$' for beta in beta_list]
        for control_num, i in zip(control_num_list, range(np.size(control_num_list))):
            for beta, j in zip(beta_list, range(np.size(beta_list))):
                if network_type == '2D':
                    ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
                else:
                    ratio_file = ratio_des + f'/d={d}/beta={beta}/netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'

                ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:1000, 1]))
                ratio_mean = np.mean(ratio)
                ratio_complete = np.sum(ratio == 1)/np.size(ratio)
                ave_list[i, j] = ratio_mean
                complete_list[i, j] = ratio_complete

    elif np.size(N_list) > 1:
        ave_list = np.zeros((np.size(control_num_list), np.size(N_list)))
        complete_list = np.zeros((np.size(control_num_list), np.size(N_list)))
        max_list = np.zeros((np.size(control_num_list), np.size(N_list)))
        beta = beta_list[0]
        labels = [f'$N={N}$' for N in N_list]
        for control_num, i in zip(control_num_list, range(np.size(control_num_list))):
            for N, j in zip(N_list, range(np.size(N_list))):
                if betaeffect:
                    ratio_des += f'beta={beta}/'
                else:
                    ratio_des += f'edgewt={beta}/'
                ratio_file = ratio_des + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'
                ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:1000, 1]))
                ratio_mean = np.mean(ratio)
                ratio_complete = np.sum(ratio == 1)/np.size(ratio)
                ave_list[i, j] = ratio_mean
                complete_list[i, j] = ratio_complete

    else:
        ave_list = np.zeros((np.size(control_num_list)))
        complete_list = np.zeros((np.size(control_num_list)))
        max_list = np.zeros((np.size(control_num_list)))
        N = N_list[0]
        beta = beta_list[0]
        labels= f'N={N}'
        for control_num, i in zip(control_num_list, range(np.size(control_num_list))):
            ratio_file = ratio_des + f'/d={d}/beta={beta}/netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'

            ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:, 1]))
            ratio_mean = np.mean(ratio)
            ratio_complete = np.sum(ratio == 1)/np.size(ratio)
            ratio_max = np.max(ratio)
            ave_list[i] = ratio_mean
            complete_list[i] = ratio_complete
            max_list[i] = ratio_max
    if plot_type == 'mean':
        feature = ave_list
        ylabel = '$\\langle R \\rangle$'
    elif plot_type == 'complete':
        feature = complete_list
        ylabel = '$R=1$'
    elif plot_type == 'max':
        feature = max_list
        ylabel = '$R_{max}$'
    plt.plot(control_num_list, feature, '-o', linewidth=lw, alpha = alpha, label=labels)
    #plt.plot(np.array(control_num_list)/np.sqrt(N), feature, linewidth=lw, alpha = alpha)
    plt.xlabel('$M$', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    if np.size(beta_list)>1 or np.size(N_list)>1:
        plt.legend(labels, frameon=False, fontsize = legendsize)



    #plt.show()

def control_value_influence(dynamics, network_type, N, beta_list, control_num_list, control_value_list, plot_type):
    """TODO: Docstring for beta_influence.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :beta_set: TODO
    :control_num: TODO
    :control_value: TODO
    :returns: TODO

    """
    ratio_des = '../data/' +  dynamics.__name__ + '_' + network_type 
    if np.size(beta_list) > 1:
        ave_list = np.zeros((np.size(control_value_list), np.size(beta_list)))
        complete_list = np.zeros((np.size(control_value_list), np.size(beta_list)))
        control_num = control_num_list[0]
        labels = [f'$\\beta={beta}$' for beta in beta_list]
        for control_value, i in zip(control_value_list, range(np.size(control_value_list))):
            for beta, j in zip(beta_list, range(np.size(beta_list))):
                ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
                ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:1000, 1]))
                ratio_mean = np.mean(ratio)
                ratio_complete = np.sum(ratio == 1)/np.size(ratio)
                ave_list[i, j] = ratio_mean
                complete_list[i, j] = ratio_complete

    elif np.size(control_num_list) > 1:
        ave_list = np.zeros((np.size(control_value_list), np.size(control_num_list)))
        complete_list = np.zeros((np.size(control_value_list), np.size(control_num_list)))
        beta = beta_list[0]
        labels = [f'$M={control_num}$' for control_num in control_num_list]
        for control_value, i in zip(control_value_list, range(np.size(control_value_list))):
            for control_num, j in zip(control_num_list, range(np.size(control_num_list))):
                ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
                ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:1000, 1]))
                ratio_mean = np.mean(ratio)
                ratio_complete = np.sum(ratio == 1)/np.size(ratio)
                ave_list[i, j] = ratio_mean
                complete_list[i, j] = ratio_complete

    else:
        ave_list = np.zeros((np.size(control_value_list)))
        complete_list = np.zeros((np.size(control_value_list)))
        control_num = control_num_list[0]
        beta = beta_list[0]
        for control_value, i in zip(control_value_list, range(np.size(control_value_list))):
            ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
            ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:, 1]))
            ratio_mean = np.mean(ratio)
            ratio_complete = np.sum(ratio == 1)/np.size(ratio)
            ave_list[i] = ratio_mean
            complete_list[i] = ratio_complete
    if plot_type == 'mean':
        feature = ave_list
        ylabel = 'average ratio'
    elif plot_type == 'complete':
        feature = complete_list
        ylabel = 'complete ratio'

    plt.plot(control_value_list, feature, linewidth=lw, alpha = alpha)
    plt.xlabel('$v$', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    if np.size(beta_list)>1 or np.size(control_num_list)>1:
        plt.legend(labels, frameon=False, fontsize = legendsize)

    #plt.show()

def dynamics_simulation(dynamics, network_type, d, network_seed, control_num_list, N, beta, betaeffect, control_value, color, label):
    """TODO: Docstring for beta_influence.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :beta_set: TODO
    :control_num: TODO
    :control_value: TODO
    :returns: TODO

    """
    ratio_dir = '../data/' +  dynamics.__name__ + '/' + network_type + f'/c={d}'
    if betaeffect:
        ratio_dir = ratio_dir + f'/beta={beta}/'
    else:
        ratio_dir = ratio_dir + f'/edgewt={beta}/'

    r_H_mean = np.zeros((np.size(control_num_list)))
    r_L_mean = np.zeros((np.size(control_num_list)))
    for control_num, i in zip(control_num_list, range(np.size(control_num_list))):
        if network_type == '2D':
            ratio_file = ratio_dir + f'N={N}_control_num={control_num}_value={control_value}.csv'
        else:
            ratio_file = ratio_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'

        ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:, 1]))
        r_H = ratio[ratio>=0.5]
        r_L = ratio[ratio<0.5]
        if len(r_H):
            r_H_mean[i] = np.mean(r_H)
        else:
            r_H_mean[i] = np.mean(r_L)
        if len(r_L):
            r_L_mean[i] = np.mean(r_L)
        else:
            r_L_mean[i] = np.mean(r_H)

    feature = np.vstack((r_H_mean, r_L_mean)).transpose()
    plt.plot(control_num_list, r_H_mean, 'd--', linewidth=lw, alpha = alpha*0.8, color = color, label=label)
    plt.plot(control_num_list, r_L_mean, 'd-.', linewidth=lw, alpha = alpha*0.8, color = color)
    #plt.plot(np.array(control_num_list)/np.sqrt(N), feature, linewidth=lw, alpha = alpha)


    #plt.show()

def percolation_theory(network_type, c, beta, betaeffect, f, color, label):
    """TODO: Docstring for percolation_transition.

    :network_type: TODO
    :f_list: TODO
    :c: TODO
    :k: TODO
    :returns: TODO

    """
    active_dir = '../data/percolation/' + network_type +  f'/c={c}/'
    if betaeffect:
        active_dir = active_dir + f'beta={beta}/'
    else:
        active_dir = active_dir + f'edgewt={beta}/'

    active_data = np.array(pd.read_csv(active_dir+'theory.csv', header=None).iloc[:, :])
    f_list = active_data[:, 0]
    active_fraction = active_data[:, 1]

    if f != None:
        index =  (f_list>=f[0]) & (f_list<=f[1])
        f_list = f_list[index]
        active_fraction =active_fraction[index]

    sort_order = np.argsort(f_list)
    f_list = f_list[sort_order]
    active_fraction = active_fraction[sort_order]
    plt.plot(f_list, active_fraction, '-', linewidth=lw, alpha = alpha*0.8, color=color, label=label)
    #plt.show()

def percolation_simulation(network_type, c, beta, betaeffect, f_list, network_seed, N, color, label):
    """TODO: Docstring for percolation_simulation.

    :network_type: TODO
    :c: TODO
    :k: TODO
    :network_seed: TODO
    :N: TODO
    :returns: TODO

    """
    
    active_dir = '../data/percolation/' + network_type +  f'/c={c}/'
    if betaeffect:
        active_dir = active_dir + f'beta={beta}/'
    else:
        active_dir = active_dir + f'edgewt={beta}/'

    r_L_mean = np.zeros(np.size(f_list))
    r_H_mean = np.zeros(np.size(f_list))
    for f, i in zip(f_list, range(np.size(f_list))):
        simu_data = np.array(pd.read_csv(active_dir + f'netseed={network_seed}_N={N}_f={f}.csv', header=None).iloc[:, :])
        active_seed = simu_data[:, 0]
        active_fraction = simu_data[:, 1]
        r_H = active_fraction[active_fraction>=0.5]
        r_L = active_fraction[active_fraction<0.5]
        if len(r_H):
            r_H_mean[i] = np.mean(r_H)
        else:
            r_H_mean[i] = np.mean(r_L)
        if len(r_L):
            r_L_mean[i] = np.mean(r_L)
        else:
            r_L_mean[i] = np.mean(r_H)
    plt.plot(f_list, r_H_mean, 'o--', linewidth=lw, alpha = alpha*0.8, color=color, label=label)
    plt.plot(f_list, r_L_mean, 'o-.', linewidth=lw, alpha = alpha*0.8, color=color)
    #plt.show()


    return r_H_mean, r_L_mean

def plot_dyn_per_theory(network_type, c, f, N, beta, betaeffect, control_value):
    """TODO: Docstring for plot_dyn_per_theory.

    :network_type: TODO
    :c: TODO
    :k: TODO
    :f: TODO
    :N: TODO
    :: TODO
    :returns: TODO

    """

    percolation_theory(network_type, c, beta, betaeffect, [f[0], f[-1]], 'black', 'theory')
    percolation_simulation(network_type, c, beta, betaeffect, f, network_seed, N, 'tab:blue', 'activating')
    dynamics_simulation(dynamics, network_type, c, network_seed, f, N, beta, betaeffect, control_value, 'tab:red', 'dynamics')
    plt.xlabel('$f$', fontsize=fontsize)
    plt.ylabel('$\\langle R \\rangle$', fontsize=fontsize)
    plt.subplots_adjust(left=0.15, right=0.85, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right',bbox_to_anchor=(1.25, 0.)  )
    plt.locator_params(axis='x', nbins=5)
    plt.show()

def percolation_finite_size(network_type, c, k, f_list, N_list, network_seed):
    """TODO: Docstring for finite_size.

    :network_type: TODO
    :c: TODO
    :k: TODO
    :network_seed: TODO
    :: TODO
    :returns: TODO

    """
    colors = itertools.cycle(('tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink', 'grey', 'tab:cyan'))

    for N, f in zip(N_list, f_list):
        percolation_simulation(network_type, c, k, f, network_seed, N, next(colors),  '$N$ = $10^{{{}}}$'.format(int(np.log10(N))))
    percolation_theory(network_type, c, k, [f_list[-1][0], f_list[-1][-1]], 'black', 'theory')

    plt.xlabel('$f$', fontsize=fontsize)
    plt.ylabel('$\\langle R \\rangle$', fontsize=fontsize)
    plt.subplots_adjust(left=0.15, right=0.85, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right',bbox_to_anchor=(1.25, 0.)  )
    plt.locator_params(axis='x', nbins=5)
    plt.show()
    return None


def threshold_heatmap(dynamics, beta):
    """TODO: Docstring for threshold_sL_sH.

    :arg1: TODO
    :returns: TODO

    """
    des_file = '../data/' + dynamics + f'/threshold/xs_sL_sH_beta={beta}.csv'
    data = np.array((pd.read_csv(des_file, header=None).iloc[:, :]))
    sL = data[1:, 0]
    sH = data[0, 1:]
    xs_list = data[1:, 1:]
    ytick_num = 5
    yticks = np.round(np.linspace(0, len(sL), ytick_num, dtype=int), 2)
    yticklabels = np.round(np.linspace(sL[0], sL[-1], ytick_num), 1)
    y_ratio = (yticks[1] - yticks[0]) / (yticklabels[1] - yticklabels[0])
    xtick_num = 5
    xticks = np.linspace(0, len(sH), xtick_num, dtype=int)
    xticklabels = np.round(np.linspace(sH[0], sH[-1], xtick_num), 2)
    x_ratio = (xticks[1] - xticks[0]) / (xticklabels[1] - xticklabels[0])
    ax = sns.heatmap(xs_list, cmap='RdBu_r', xticklabels=xticklabels, yticklabels=yticklabels,  linewidths=0)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(xticklabels, fontsize=15)
    ax.set_yticklabels(yticklabels, fontsize=15)
    ax.invert_yaxis()
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=15)

    xs_sL = xs_list[:, 0]
    xs_sH = xs_list[0]
    sL_c = sL[np.where(xs_sL > threshold)[0][0]]
    sH_c = sH[np.where(xs_sH > threshold)[0][0]]
    plt.plot([sH_c * x_ratio, 0], [0, sL_c*y_ratio], '-', color='k', linewidth = 2)
    plt.xlabel('$s_H$', fontsize=fs)
    plt.ylabel('$s_L$', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
    return None

def threshold_line(dynamics):
    """TODO: Docstring for threshold_sL_sH.

    :arg1: TODO
    :returns: TODO

    """
    des_file = '../data/' + dynamics + f'/threshold/sL_sH_beta.csv'
    data = np.array((pd.read_csv(des_file, header=None).iloc[:, :]))
    beta_list, xL_list, xH_list, sL_list, sH_list = data.transpose()
    for i, beta in enumerate(beta_list):
        plt.plot([0, sL_list[i]], [sH_list[i], 0], linewidth=2.5, label=f'beta={beta}')
    plt.xlabel('$s_H$', fontsize=fs)
    plt.ylabel('$s_L$', fontsize=fs)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize)

    return None

def nH_degree(dynamics, network_type, N, network_seed, d, weight_list):
    """original dynamics N species interaction.

    :x: N dynamic variables, 1 * N vector 
    :t: time 
    :N: the number of interacting variables 
    :index: the index of non-zero element of adjacency matrix A, N * N matrix 
    :neighbor: the degree of each node 
    :A_interaction: non-zero element of adjacency matrix A, 1 * N vector
    :returns: derivative of x 

    """
    for weight in weight_list:
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, weight, 0, network_seed, d)
        beta, _ = np.round(betaspace(A, [0]), 5)
        des_file = '../data/' + dynamics + f'/threshold/beta={beta}_w={weight}.csv'
        data = np.array((pd.read_csv(des_file, header=None).iloc[:, :]))
        degree_list, neighbor_H = data.transpose()
        index = np.where(neighbor_H >= 0)[0]

        plt.plot(degree_list[index], neighbor_H[index], '--', label=f'w={weight}')
    plt.xlabel('degree $k$', fontsize=fontsize)
    plt.ylabel('$n_H$', fontsize=fontsize)
    plt.subplots_adjust(left=0.15, right=0.8, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize, loc='upper left',bbox_to_anchor=(0.9, 1.)  )

    #plt.show()

    return None

def percolation_activation_dynamics(dynamics, network_type, N, network_seed, d, weight, control_seed):
    """TODO: Docstring for percolation_activation_dynamics.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :network_seed: TODO
    :d: TODO
    :weight: TODO
    :returns: TODO

    """
    activation_des = '../data/' + dynamics + '/' + network_type + '/percolation_activation/' 
    activation_file = activation_des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}.csv'
    dynamics_des = '../data/' + dynamics + '/' + network_type + '/multi_dynamics/'
    dynamics_file = dynamics_des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_multi_xs.csv'
    activation_data = np.array(pd.read_csv(activation_file, header=None).iloc[:, :])
    dynamics_data = np.array(pd.read_csv(dynamics_file, header=None).iloc[:, :])
    f = activation_data[:, 0]
    active_state = activation_data[:, 1:]
    xs = dynamics_data[:, 1:]
    return f, active_state, xs

    
    


dynamics = 'mutual'
network_type = 'ER'
network_seed = 0
c = 16
k = 7

#percolation_theory(network_type, c, k, None, 'k', '')

f_list = np.round(np.hstack((np.arange(0.03, 0.0501, 0.001), np.arange(0.055, 0.075, 0.005), np.array([0.071, 0.075]))), 3)
f_list = np.round(np.arange(0.0002, 0.0041, 0.0002), 4)

f_1 = np.round(np.arange(0.03, 0.07, 0.002), 3)
f_2 = np.sort(np.round(np.hstack((np.arange(0.03, 0.07, 0.002), np.arange(0.041, 0.051, 0.002))), 3))
f_3 = np.sort(np.round(np.hstack((np.arange(0.03, 0.07, 0.002), np.arange(0.041, 0.051, 0.002))), 3))
f_4 = np.sort(np.round(np.hstack((np.arange(0.044, 0.045, 0.0001), np.arange(0.03, 0.075, 0.01))),  4))
f_5 = [0.03, 0.07]



f_list = np.sort(np.round(np.hstack((np.arange(0, 0.21, 0.01), np.arange(0.16, 0.18, 0.001))), 3))
f_list = np.round(np.arange(0.16, 0.17, 0.001), 3)
N = 100000
#percolation_simulation(network_type, c, k, f_list, network_seed, N, 'k', '')


d = 2000
network_seed = 0
control_num_list = np.round(np.hstack((np.arange(0.0002, 0.0042, 0.0002))), 4)
beta = 0.2
betaeffect = 0
control_value = 1


#dynamics_simulation(dynamics, network_type, d, network_seed, control_num_list, N, beta, betaeffect, control_value, 'b', '')

f_list = np.round(np.arange(0.03, 0.054, 0.002), 3)
c = 4
k = 4
beta = 0.1
N = 10000
f_list = np.round(np.arange(0.01, 0.07, 0.002), 3)

c = 16
k = 7
beta = 0.1
N = 10000
f_list = np.round(np.arange(0.16, 0.24, 0.001), 3)
f_list = np.sort(np.round(np.hstack((np.arange(0.1, 0.21, 0.01), np.arange(0.16, 0.18, 0.005))), 3))
f_list = np.round(np.arange(0.030, 0.0522, 0.002), 3)
f_list = np.round(np.arange(0.01, 0.06, 0.002), 3)

#plot_dyn_per_theory(network_type, c, f_list, N, beta, betaeffect, control_value)

dynamics = 'mutual'
beta = 0.1
threshold = 5.0

#threshold_heatmap(dynamics, beta)
#threshold_line(dynamics)
weight_list = [0.05, 0.1, 0.2]
d = 8000
N = 1000
#nH_degree(dynamics, network_type, N, network_seed, d, weight_list)

d = 4000
weight = 0.1
control_seed = 0
f, active_state, xs = percolation_activation_dynamics(dynamics, network_type, N, network_seed, d, weight, control_seed)
