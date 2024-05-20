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

fontsize = 22
ticksize = 18
legendsize = 18
lw = 2.5
alpha = 0.8
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1


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
            #evolution_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/d={d}/edgewt={beta}/evolution/'
            evolution_dir = '../data/result1201/' +  dynamics.__name__ + '_' + network_type + f'/d={d}/edgewt={beta}/evolution/'
        evolution_des = evolution_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}_control_seed={control_seed}.npy'
        heatmap_des = evolution_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}/control_seed={control_seed}/'
        G = nx.from_numpy_matrix(A)
        position = nx.spring_layout(G, seed=0 )

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

            #nx.draw(G, pos=position, cmap=cmap, node_color=data_snap, vmin=vmin, vmax=vmax)

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
        fig.savefig(heatmap_des + str(int(i/plot_interval)) + '.svg', format="svg")

        plt.close()
        #plt.show()
    return None

def plot_ratio_distribution(dynamics, network_type, d, network_seed, beta, betaeffect, N, control_num, control_value):
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

    if network_type == '2D':
        ratio_dir = '../data/' +  dynamics.__name__ + '_' + network_type 
    else:
        ratio_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/d={d}'

    if betaeffect:
        ratio_dir = ratio_dir + f'/beta={beta}/'
    else:
        ratio_dir = ratio_dir + f'/edgewt={beta}/'

    if network_type == '2D':
        ratio_des = ratio_dir + f'N={N}_control_num={control_num}_value={control_value}.csv'
    else:
        ratio_des = ratio_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'



    ratio = np.array(list(pd.read_csv(ratio_des, header=None).iloc[:, 1]))
    bins = np.arange(0, 1.01, 0.01)

    frequency, bins, patches = plt.hist(ratio, bins=bins, density=True)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Transition ratio $R$', fontsize=fontsize)
    plt.ylabel('Frequency %', fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize)

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
    ratio_des = '../data/' +  dynamics.__name__ + '_' + network_type 
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
                if network_type == '2D':
                    ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
                else:
                    ratio_file = ratio_des + f'/d={d}/beta={beta}/netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'

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
            if network_type == '2D':
                ratio_file = ratio_des + f'/beta={beta}/N={N}_control_num={control_num}_value={control_value}.csv'
            else:
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

def R_HL_M(dynamics, network_type, d, network_seed, control_num_list, N, beta, betaeffect, control_value, plot_type):
    """TODO: Docstring for beta_influence.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :beta_set: TODO
    :control_num: TODO
    :control_value: TODO
    :returns: TODO

    """
    if network_type == '2D':
        ratio_dir = '../data/' +  dynamics.__name__ + '_' + network_type 
    else:
        ratio_dir = '../data/' +  dynamics.__name__ + '_' + network_type + f'/d={d}'
    if betaeffect:
        ratio_dir = ratio_dir + f'/beta={beta}/'
    else:
        ratio_dir = ratio_dir + f'/edgewt={beta}/'



    ave_list = np.zeros((np.size(control_num_list)))
    complete_list = np.zeros((np.size(control_num_list)))
    max_list = np.zeros((np.size(control_num_list)))
    r_H_mean = np.zeros((np.size(control_num_list)))
    r_L_mean = np.zeros((np.size(control_num_list)))
    for control_num, i in zip(control_num_list, range(np.size(control_num_list))):
        if network_type == '2D':
            ratio_file = ratio_dir + f'N={N}_control_num={control_num}_value={control_value}.csv'
        else:
            ratio_file = ratio_dir + f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'

        ratio = np.array(list(pd.read_csv(ratio_file,header=None).iloc[:, 1]))
        ratio_mean = np.mean(ratio)
        ratio_complete = np.sum(ratio == 1)/np.size(ratio)
        ratio_max = np.max(ratio)
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
    elif plot_type == 'HL_mean':
        feature = np.vstack((r_H_mean, r_L_mean)).transpose()
        ylabel = '$\\langle R \\rangle$'
    plt.plot(control_num_list, r_H_mean, '--', linewidth=lw, alpha = alpha*0.8)
    plt.plot(control_num_list, r_L_mean, '-.', linewidth=lw, alpha = alpha*0.8)
    #plt.plot(np.array(control_num_list)/np.sqrt(N), feature, linewidth=lw, alpha = alpha)
    plt.xlabel('$M$', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.98, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(['$R_H$', '$R_L$'], frameon=False, fontsize = legendsize)


    #plt.show()




dynamics = mutual_multi
network_type = '2D'
network_type = 'ER'
network_seed = 0
d = 10
N = 10
edgewt = 0.2
beta = 0.2
betaeffect = 0
control_num = 3
control_value = 1
control_seed = 15
plot_interval = 50

heatmap(dynamics, network_type, d, network_seed, beta, betaeffect, N, control_num, control_value, control_seed, plot_interval, dt=0.01, linewidth=0)
N_list = [900]
control_value = 1
control_num_list = [5, 20,25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100]
control_num_list = [10, 50, 60, 70, 80, 90, 100, 105, 110, 115, 120, 125, 130, 135, 140, 150, 170, 200]
control_num_list = [10, 20, 21, 22, 23, 25, 30, 35, 40, 45, 50, 55, 60, 61, 62, 63, 65, 70, 80, 90, 100, 200]
control_num_list = [5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,  90, 100, 200]
control_num_list = [10, 20, 30, 40, 50, 60, 61, 62, 63, 65, 70, 75, 80, 85, 90, 95, 100, 110, 113, 114, 115, 120, 130, 140, 150, 200]
N = 900

beta_list =  [round(beta, 2) if round(beta, 2)%1>0 else round(beta) for beta in np.arange(0.4, 1.4, 0.02)]
beta_list = [0.4, 0.6, 0.8, 1, 1.2, 1.4]
beta_list = [1]


beta_list = [0.2]
control_num_list = [4, 5, 6, 10, 15, 19, 20, 30, 40, 50, 60, 70, 100]
control_num_list = [1, 2, 3, 4, 5, 10]
control_num_list = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800]
control_num_list = [10, 50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 700, 800]
control_num_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]




d= 1800
beta = 0.1
control_num_list = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800]





beta = 0.3
d = 7200
control_num_list =np.arange(1, 10, 1).astype(int).tolist()


beta = 0.1
d = 7200
control_num_list = np.hstack((np.arange(10, 100, 10), np.arange(100, 250, 50))).astype(int).tolist()

d= 1800
beta = 0.3
control_num_list = np.hstack((np.arange(1, 10, 1), np.arange(10, 60, 5), np.arange(100, 300, 100))).astype(int).tolist()

d = 1800
beta = 0.2
control_num_list = [10, 20, 30, 31, 32, 33, 34, 35, 40, 50, 60, 61, 62, 63, 65, 70, 100, 110, 120, 130, 150, 200]

beta = 0.1
d = 3600
control_num_list = np.hstack(([10, 50], np.arange(100, 200, 10), np.arange(200, 300, 100))).astype(int).tolist()


beta = 0.2
d = 3600
control_num_list = np.hstack((np.arange(4, 20, 1), np.arange(20, 70, 10), [100, 200])).astype(int).tolist()


beta = 0.3
d = 3600
control_num_list = np.hstack((np.arange(1, 10, 1), [50, 100, 200])).astype(int).tolist()

beta = 0.2
d = 7200
control_num_list = np.hstack((np.arange(1, 5, 1), [5, 10, 50, 100, 200])).astype(int).tolist()

N = 10000
beta = 0.2
d = 20000


control_num = 0.042
control_seed = 0

betaeffect = 0

#heatmap(dynamics, network_type, d, network_seed, beta, betaeffect, N, control_num, control_value, control_seed, plot_interval)

#ratio = plot_ratio_distribution(dynamics, network_type, d, network_seed, beta, betaeffect, N, control_num, control_value)

plot_type = 'complete'
plot_type = 'max'
plot_type = 'mean'
plot_type ='HL_mean'
#beta_influence(dynamics, network_type, N_list, beta_list, control_num_list, control_value, plot_type)

#control_num_influence(dynamics, network_type, d, network_seed, N_list, beta_list, control_num_list, control_value, plot_type)
#control_value_list = [0.5, 0.6, 0.7, 0.75, 0.8,0.85, 0.9, 1, 1.1, 1.2]
#control_num_list = [10, 20, 30]
#control_value_influence(dynamics, network_type, N, beta_list, control_num_list, control_value_list, plot_type)
#R_HL_M(dynamics, network_type, d, network_seed, control_num_list, N, beta, betaeffect, control_value, plot_type)
#plt.show()
