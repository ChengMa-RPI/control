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
from cycler import cycler
import matplotlib as mpl
from collections import Counter

fs = 22
ticksize = 16
legendsize = 18
lw = 2.5
alpha = 0.8
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-', '--']))


class data_plot:

    """plot data"""
    def __init__(self, network_type, N, beta, betaeffect, network_seed, d, control_constant, control_seed, method, interval):
        self.network_type, self.N, self.beta, self.betaeffect, self.network_seed, self.d, self.control_seed, self.method, self.interval = network_type, N, beta, betaeffect, network_seed, d, control_seed, method, interval
        if method == 'KNN':
            des = '../data/' + dynamics + '/' + network_type + f'/KNN/'
        elif method == 'firstNN':
            des = '../data/' + dynamics + '/' + network_type + f'/firstNN/'
        elif method == 'KNN_degree':
            des = '../data/' + dynamics + '/' + network_type + f'/KNN_degree_' + space + f'={degree_interval}/'
        elif method == 'KNN_connectcontrol':
            des = '../data/' + dynamics + '/' + network_type + f'/KNN_subgroup_interval={interval}/'
        self.des = des 


    def data_reduction(self):
        des = self.des 
        network_type, N, beta, betaeffect, network_seed, d, control_seed, method, interval = self.network_type, self.N, self.beta, self.betaeffect, self.network_seed, self.d, self.control_seed, self.method, self.interval 
        if betaeffect == 0:
            des_reduction_file = des + f'N={N}_d={d}_netseed={network_seed}_wt={beta}_controlseed={control_seed}.csv'
        else:
            des_reduction_file = des + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}.csv'

        data = np.array(pd.read_csv(des_reduction_file, header=None, names=np.arange(10000)).iloc[:, :])
        data_size = length_groups_max = np.max(np.sum(1 - np.isnan(data), 1) )  
        control_num_list = data[:, 0]
        length_groups_max = int(np.sqrt(data_size) - 1)
        x_eff = np.ones((len(control_num_list), length_groups_max)) * (-1)
        xs_reduction = np.ones((len(control_num_list), length_groups_max)) * (-1)
        for i, control_num in enumerate(control_num_list):
            data_i = data[i]
            data_i = data_i[np.logical_not(np.isnan(data_i))]
            length_groups = int(np.sqrt(np.size(data_i) ) - 1)
            A_reduction = data_i[1: length_groups**2+1]
            A_reduction_diagonal = np.vstack(([A_reduction[i*length_groups+i] for i in range(length_groups)])).transpose()
            x_eff[i, :length_groups] = data_i[length_groups**2+1:length_groups**2+length_groups+1]
            xs_reduction[i, :length_groups] = data_i[length_groups**2+length_groups+1:length_groups**2+2*length_groups+1]

        self.x_eff, self.xs_reduction = x_eff, xs_reduction
        self.control_num_list = control_num_list
        self.length_groups_max = length_groups_max

    def data_multi(self):
        """TODO: Docstring for data_multi.
        :returns: TODO

        """
        des = self.des
        self.network_type, self.N, self.beta, self.betaeffect, self.network_seed, self.d, self.control_seed, self.method, self.interval = network_type, N, beta, betaeffect, network_seed, d, control_seed, method, interval
        if betaeffect == 0:
            des_multi_file = des + f'N={N}_d={d}_netseed={network_seed}_wt={beta}_controlseed={control_seed}_multi_xs.csv'
        else:
            des_multi_file = des + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'

        data = np.array(pd.read_csv(des_multi_file, header=None).iloc[:, :])
        control_num_list = data[::4, 0]
        node_index = data[::4, 1:]
        node_group_mapping = data[1:][::4, 1:]
        xs_multi = data[2:][::4, 1:]
        xs_decouple = data[3:][::4, 1:]
        self.node_index = node_index
        self.node_group_mapping = node_group_mapping
        self.xs_multi = xs_multi
        self.xs_decouple = xs_decouple
        self.control_num_list = control_num_list

    def plot_xeff_f(self, plot_num):
        """TODO: Docstring for plot_xeff_f.

        :arg1: TODO
        :returns: TODO

        """
        self.network_type, self.N, self.beta, self.betaeffect, self.network_seed, self.d, self.control_seed, self.method, self.interval = network_type, N, beta, betaeffect, network_seed, d, control_seed, method, interval

        control_num_list = self.control_num_list
        length_groups_max = self.length_groups_max 
        y_theory = self.xs_reduction
        y_simulation = self.x_eff
        "not plot the control set"
        if plot_num == 'all':
            plot_length = length_groups_max
        elif plot_num == 'major':
            plot_length = 2
        for i in range(1, plot_length):
            index_valid = np.where((y_theory[:, i]) >0)[0]
            plt.plot(control_num_list[index_valid], y_theory[index_valid, i], '-', linewidth=lw, label='theory' + str(i))
            plt.plot(control_num_list[index_valid], y_simulation[index_valid, i], '--', linewidth=lw, label='simulation' + str(i))

        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.subplots_adjust(left=0.2, right=0.75, wspace=0.25, hspace=0.25, bottom=0.18, top=0.80)
        plt.locator_params('x', nbins=5)
        plt.xlabel('$f$', fontsize=fs)
        plt.ylabel('$x_s$', fontsize=fs)
        #plt.legend(fontsize=13, frameon=False) 
        plt.legend(fontsize=13, frameon=False, loc=1, bbox_to_anchor=(1.5,1.2) )
        save_des = '../report/report122121/' + f'N={N}_d={d}_netseed={network_seed}_wt={beta}_controlseed={control_seed}_' + method + '_'
        if method == 'KNN_connectcontrol':
            save_des += f'interval={interval}_'
        if plot_num == 'all':
            plt.savefig(save_des + 'xeff_comp.png')
            pass
        elif plot_num == 'major':
            plt.savefig(save_des + 'xeff_comp_major.png')
            pass
        plt.close('all')
        #plt.show()

    def ratio_f(self, ratio_denominator, survival_criteria):
        """TODO: Docstring for plot_ratio_f.

        :f: TODO
        :returns: TODO

        """
        control_num_list = self.control_num_list
        xs_multi = self.xs_multi
        xs_decouple = self.xs_decouple
        node_group_mapping = self.node_group_mapping
        node_index = self.node_index
        xs_reduction = self.xs_reduction
        x_eff = self.x_eff
        N_actual = np.size(node_index, 1)

        survival_num_multi_list = np.zeros((len(control_num_list)))
        survival_num_reduction_list = np.zeros((len(control_num_list)))
        survival_num_decouple_list = np.zeros((len(control_num_list)))

        for i, control_num in enumerate(control_num_list):
            each_group_length = Counter(node_group_mapping[i])
            survival_num_multi_list[i] = sum(xs_multi[i] > survival_criteria)
            survival_num_decouple_list[i] = sum(xs_decouple[i] > survival_criteria)
            survival_group = np.where(xs_reduction[i] > survival_criteria)[0]
            survival_num_reduction_list[i] = sum([each_group_length[j] for j in survival_group])
        if ratio_denominator == 'all_node':
            denominator  = N_actual
        elif ratio_denominator == 'uncontrol_node':
            denominator = N_actual - np.array(N_actual * control_num_list, int)
        survival_rate_multi_list = survival_num_multi_list / denominator
        survival_rate_decouple_list = survival_num_decouple_list / denominator
        survival_rate_reduction_list = survival_num_reduction_list / denominator

        self.survival_rate_multi_list = survival_rate_multi_list
        self.survival_rate_decouple_list = survival_rate_decouple_list
        self.survival_rate_reduction_list = survival_rate_reduction_list

    def plot_ratio_f(self, ratio_denominator):
        """TODO: Docstring for plot_ratio_f.
        :returns: TODO

        """
        control_num_list = self.control_num_list
        survival_rate_multi_list =  self.survival_rate_multi_list 
        survival_rate_decouple_list = self.survival_rate_decouple_list
        survival_rate_reduction_list = self.survival_rate_reduction_list
        
        if ratio_denominator == 'all_node':
            R_label = 'ratio $R_2$'
        elif ratio_denominator == 'uncontrol_node':
            R_label = 'ratio $R_1$'

        plt.plot(control_num_list, survival_rate_multi_list, label='simulation', linestyle='-', linewidth=lw, color='#fc8d62')
        plt.plot(control_num_list, survival_rate_reduction_list, label='reduction', linestyle='-', linewidth=lw, color='#66c2a5')
        plt.plot(control_num_list, survival_rate_decouple_list, label='decouple', linestyle='-', linewidth=lw, color='#e78ac3')
        plt.legend(fontsize=legendsize, frameon=False)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
        plt.xlabel('control_num $f$', fontsize=fs)
        plt.ylabel(R_label, fontsize=fs)
        save_des = '../report/report092821/' + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_' + method + '_'
        if method == 'KNN_connectcontrol':
            save_des += f'interval={interval}_' 
        plt.savefig(save_des + ratio_denominator + '_r_f.png')
        plt.close('all')
        #plt.show()



def calculate_distr_ratio(network_type, N, weight, network_seed, d, control_constant, control_seed_list, method, interval, ratio_denominator, survival_criteria):
    """TODO: Docstring for plot_dis_ratio.
    :returns: TODO

    """
    survival_rate_multi_diffseed = []
    survival_rate_decouple_diffseed = []
    survival_rate_reduction_diffseed = []
    for control_seed in control_seed_list:
        data = data_plot(network_type, N, weight, network_seed, d, control_constant, control_seed, method, interval)
        data.data_reduction()
        data.data_multi()
        data.ratio_f(ratio_denominator, survival_criteria)

        control_num_list = data.control_num_list
        survival_rate_multi_diffseed.append( data.survival_rate_multi_list )
        survival_rate_reduction_diffseed.append( data.survival_rate_reduction_list )
        survival_rate_decouple_diffseed.append( data.survival_rate_decouple_list )
    survival_rate_multi_diffseed = np.vstack((survival_rate_multi_diffseed))
    survival_rate_reduction_diffseed = np.vstack((survival_rate_reduction_diffseed))
    survival_rate_decouple_diffseed = np.vstack((survival_rate_decouple_diffseed))

    return survival_rate_multi_diffseed, survival_rate_reduction_diffseed, survival_rate_decouple_diffseed

def plot_ratio_dis(network_type, N, weight, network_seed, d, control_constant, control_seed_list, method, interval, ratio_denominator, survival_criteria, control_num_list, curve_control_num_list):
    """TODO: Docstring for plot_ratio_dis.

    :arg1: TODO
    :returns: TODO

    """
    survival_rate_multi_diffseed, survival_rate_reduction_diffseed, survival_rate_decouple_diffseed = calculate_distr_ratio(network_type, N, weight, network_seed, d, control_constant, control_seed_list, method, interval, ratio_denominator, survival_criteria)
    if ratio_denominator == 'all_node':
        R_label = 'ratio $R_2$'
    elif ratio_denominator == 'uncontrol_node':
        R_label = 'ratio $R_1$'
    for survival_type, survival_data in zip(['multi', 'reduction', 'decouple'], [survival_rate_multi_diffseed, survival_rate_reduction_diffseed, survival_rate_decouple_diffseed]):
        prob_density = []
        for i in range(np.size(survival_data, 1)):
            n, x = np.histogram(survival_data[:, i], bins=np.arange(-0.0001, 1.02, 0.02))
            bin_centers = 0.5*(x[1:]+x[:-1])
            prob_density.append(n)

        prob_density = np.vstack((prob_density)).transpose()/np.max(prob_density)
        "heatmap"
        ytick_num = 6
        yticks = np.linspace(0, np.size(prob_density, 0), ytick_num, dtype=int)
        yticklabels = np.round(np.linspace(0, 1, ytick_num), 2)
        xtick_num = 6
        xticks = np.linspace(0, np.size(prob_density, 1), xtick_num, dtype=int)
        xticklabels = np.round(np.linspace(0, 1, xtick_num), 2)
        ax = sns.heatmap(prob_density, cmap='RdBu_r', xticklabels=xticklabels, yticklabels=yticklabels,  linewidths=0)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels, fontsize=15)
        ax.set_yticklabels(yticklabels, fontsize=15)
        ax.invert_yaxis()
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        plt.xlabel('control num $f$', fontsize=fs)
        plt.ylabel(R_label, fontsize=fs)
        plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
        save_des = '../report/report092821/' + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_' + method + '_'
        if method == 'KNN_connectcontrol':
            save_des += f'interval={interval}_' 
        plt.savefig(save_des + ratio_denominator + '_' + survival_type + '_' + 'heatmap' + '_r_f.png')
        plt.close('all')

        "curve"
        for i, control_num in enumerate(curve_control_num_list):
            index = np.where(np.abs(control_num_list - control_num) < 1e-8)[0]
            plt.plot(bin_centers, prob_density[:, index], linewidth=lw, linestyle = '-', label=f'$f=${control_num}')
            plt.legend(frameon=False, fontsize = legendsize)
            plt.xticks(fontsize=ticksize)
            plt.yticks(fontsize=ticksize)
        plt.xlabel(R_label, fontsize=fs)
        plt.ylabel('prob $P$', fontsize=fs)
        plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)

        save_des = '../report/report092821/' + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_' + method + '_'
        if method == 'KNN_connectcontrol':
            save_des += f'interval={interval}_' 
        plt.savefig(save_des + ratio_denominator + '_' + survival_type + '_' + 'curve' + '_r_f.png')
        plt.close('all')
        #plt.show()

def plot_percolation_multi_dynamics(network_type, N, weight, network_seed, d, control_constant, control_seed_list, method, interval, ratio_denominator, survival_criteria):
    """TODO: Docstring for plot_percolation_multi_dynamics.

    :arg1: TODO
    :returns: TODO

    """
    if ratio_denominator == 'all_node':
        R_label = 'ratio $R_2$'
    elif ratio_denominator == 'uncontrol_node':
        R_label = 'ratio $R_1$'



    percolation_des = '../data/' + dynamics + '/' + network_type +  f'/percolation/' +  f'N={N}_d={d}_netseed={network_seed}_wt={weight}_percolation.csv'
    percolation_data = np.array(pd.read_csv(percolation_des, header=None).iloc[:, :])
    control_num_percolation = percolation_data[:, 0]
    control_num_order = np.argsort(control_num_percolation)
    ratio_percolation = percolation_data[:, 1]

    data = data_plot(network_type, N, weight, network_seed, d, control_constant, control_seed, method, interval)
    data.data_reduction()
    data.data_multi()
    data.ratio_f(ratio_denominator, survival_criteria)
    control_num_multi = data.control_num_list

    survival_rate_multi_list =  data.survival_rate_multi_list 
    if ratio_denominator == 'uncontrol_node':
        ratio_percolation = (ratio_percolation - control_num_percolation) / (1 - control_num_percolation)
    plt.plot(control_num_percolation[control_num_order], ratio_percolation[control_num_order], linewidth=lw, label='percolation')
    plt.plot(control_num_multi, survival_rate_multi_list, linewidth=lw, label='simulation')
    plt.legend(fontsize=legendsize, frameon=False)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xlabel('control_num $f$', fontsize=fs)
    plt.ylabel(R_label, fontsize=fs)
    save_des = '../report/report092821/' + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_' + 'percolation_'
    plt.savefig(save_des + ratio_denominator + '_r_f.png')
    plt.close('all')
    #plt.show()






dynamics = 'mutual'
dynamics = 'genereg'

control_seed = 0
control_constant = 5
control_num = 0.5





betaeffect = 0

interval = 1

network_type = 'SF'
N = 1000
d = [2.5, 0, 3]
network_seed = [0, 0]
beta = 0.15
space = 'log'
degree_interval = 1.5

network_type = 'ER'
N = 1000
d = 8000
network_seed = 0
beta = 0.15
space = 'linear'
degree_interval = 10




method_list = ['KNN']
ratio_denominator_list = ['all_node', 'uncontrol_node']
survival_criteria_list = [4.8, 5]
control_num_list = np.arange(0.01, 1, 0.01)

for method in method_list:

    data = data_plot(network_type, N, beta, betaeffect, network_seed, d, control_constant, control_seed, method, interval)
    data.data_reduction()
    data.data_multi()
    for plot_num in ['major']:
        #data.plot_xeff_f(plot_num)
        pass
    for ratio_denominator, survival_criteria in zip(ratio_denominator_list, survival_criteria_list):
        #data.ratio_f(ratio_denominator, survival_criteria)
        #data.plot_ratio_f(ratio_denominator)
        pass
    #plt.show()


        control_seed_list = np.arange(100).tolist()
        curve_control_num_list = [0.13, 0.14, 0.15, 0.16]

        #plot_ratio_dis(network_type, N, weight, network_seed, d, control_constant, control_seed_list, method, interval, ratio_denominator, survival_criteria, control_num_list, curve_control_num_list)

for ratio_denominator, survival_criteria in zip(ratio_denominator_list, survival_criteria_list):
    #plot_percolation_multi_dynamics(network_type, N, weight, network_seed, d, control_constant, control_seed_list, method, interval, ratio_denominator, survival_criteria)
    pass

