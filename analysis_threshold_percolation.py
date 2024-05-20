import sys
sys.path.insert(1, '/home/mac/RPI/research/')
from mutual_framework import load_data, A_from_data, Gcc_A_mat, betaspace, stable_state, network_generate, normalization_x, gif

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
import matplotlib.cm as cm

#from threshold_percolation import nH_k_sL_sH

fs = 22
ticksize = 16
legendsize = 16
lw = 2.5
alpha = 0.8
B = 0.1
C = 1
K = 5
D = 5 
E = 0.9
H = 0.1

B_gene = 1

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-', '--']))
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f', '#e5c494', '#b3b3b3', '#7fc97f', '#bebada'])  * cycler(linestyle=['-']))

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

def genereg_multi(x, t, arguments, net_arguments):
    B, = arguments
    index_i, index_j, A_interaction, cum_index = net_arguments
    sum_f = - B * x 
    sum_g = A_interaction * x[index_j]**2/(x[index_j]**2+1)
    dxdt = sum_f + np.add.reduceat(sum_g, cum_index[:-1])
    return dxdt


def mutual_1D(x, t, c, arguments):
    B, C, D, E, H, K = arguments
    dxdt = B + x * (1 - x/K) * ( x/C - 1) + c * x**2 / (D + E*x + H*x) 
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

def mutual_decouple(x, t, x_eff, w, arguments):
    """describe the derivative of x.
    set universal parameters 
    :x: the species abundance of plant network 
    :t: the simulation time sequence 
    :par: parameters  of this system
    :returns: derivative of x 

    """
    B, C, D, E, H, K = arguments
    sum_f = B + x * (1 - x/K) * ( x/C - 1)
    sum_g = w * x * x_eff / (D + E * x + H * x_eff)
    dxdt = sum_f + sum_g
    return dxdt




class data_plot:

    """plot data"""
    def __init__(self, network_type, N, beta, betaeffect, network_seed_list, d, control_seed_list, control_num, dynamics, attractor_high, attractor_low):
        self.network_type, self.N, self.beta, self.betaeffect, self.network_seed_list, self.d, self.control_seed_list, self.control_num, self.dynamics, self.attractor_high, self.attractor_low = network_type, N, beta, betaeffect, network_seed_list, d, control_seed_list, control_num, dynamics, attractor_high, attractor_low

    def sH_sL_seed(self, network_seed, control_seed):
        """TODO: Docstring for sH_sL.

        :network_seed: TODO
        :control_seed: TODO
        :returns: TODO

        """
        network_type, N, beta, betaeffect, d, control_num, dynamics, attractor_high, attractor_low =  self.network_type, self.N, self.beta, self.betaeffect, self.d, self.control_num, self.dynamics, self.attractor_high, self.attractor_low
        if betaeffect:
            des_file = '../data/' + dynamics + '/threshold/threshold_sH_sL_multi_original/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_controlnum={control_num}.csv'
        else:
            des_file = '../data/' + dynamics + '/threshold/threshold_sH_sL_multi_original/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}_controlnum={control_num}.csv'

        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        xs_multi, sH, sL = data.transpose()
        N_actual = len(xs_multi)
        plot_index = np.arange((N_actual))
        plt.scatter(sH, sL, c=xs_multi, vmin=attractor_low, vmax=attractor_high, marker='o', s=10, cmap='coolwarm' )

    def plot_sH_sL_multi(self):
        """TODO: Docstring for plot_sH_sL_multi.
        :returns: TODO

        """
        network_seed_list, conrol_seed_list = self.network_seed_list, self.control_seed_list
        for network_seed in network_seed_list:
            for control_seed in control_seed_list:
                self.sH_sL_seed(network_seed, control_seed)
        #plt.show()

    def threshold_line(self, beta):
        """TODO: Docstring for threshold_sL_sH.

        :arg1: TODO
        :returns: TODO

        """
        network_type, N, beta, betaeffect, d, control_num, dynamics, attractor_high, attractor_low =  self.network_type, self.N, self.beta, self.betaeffect, self.d, self.control_num, self.dynamics, self.attractor_high, self.attractor_low
        des_file = '../data/' + dynamics + f'/threshold/sL_sH_beta.csv'
        data = np.array((pd.read_csv(des_file, header=None).iloc[:, :]))
        beta_list, xL_list, xH_list, sL_list, sH_list = data.transpose()
        index = np.where(np.abs(beta_list - beta) < 1e-5)[0][0]
        plt.plot([0, sH_list[index]], [sL_list[index], 0], linewidth=2.5)
        plt.xlabel('$s_H$', fontsize=fs)
        plt.ylabel('$s_L$', fontsize=fs)
        plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.legend(frameon=False, fontsize = legendsize)
        save_file = '../report/report101321/' + network_type + f'_N={N}_d={d}_beta={beta}_controlnum={control_num}_threshold_multi_netseeds={len(self.network_seed_list)}_controlseeds={len(self.control_seed_list)}.png'
        plt.savefig(save_file)
        plt.close('all')
        #plt.show()
        return None

    def R_f_activation(self, network_seed, control_seed, reverse):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect
        if betaeffect:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}.csv'
        else:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}.csv'

        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        f = data[:, 0]
        active_state = data[:, 1:]
        N_actual = np.size(active_state, 1)
        R = np.sum(active_state, 1) / N_actual
        if reverse == 'reverse':
            x = 1-f
            y = 1-R
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f
            y = R
            xlabels = '$f$'
            ylabels = '$R$'
        if control_seed == 0:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5, label='activation simulation')
        else:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5)

    def R_f_activation_diffstate(self, network_seed, control_seed, reverse):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect
        if betaeffect:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_diffstate/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}.csv'
        else:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_diffstate/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}.csv'

        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        f = data[:, 0]
        active_state = data[:, 1:]
        N_actual = np.size(active_state, 1)
        R = np.sum(active_state, 1) / N_actual
        if reverse == 'reverse':
            x = 1-f
            y = 1-R
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f
            y = R
            xlabels = '$f$'
            ylabels = '$R$'
        if control_seed == 0:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5, label='activation simulation')
        else:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5)

    def R_f_activation_diffthreshold(self, network_seed, control_seed, reverse, ratio_threshold):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect
        if betaeffect:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_diffthreshold_ratio={ratio_threshold}/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}.csv'
        else:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_diffthreshold_ratio={ratio_threshold}/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}.csv'

        data = np.array(pd.read_csv(des_file, header=None).iloc[:100, :])
        f = data[:, 0]
        active_state = data[:, 1:]
        N_actual = np.size(active_state, 1)
        R = np.sum(active_state, 1) / N_actual
        if reverse == 'reverse':
            x = 1-f
            y = 1-R
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f
            y = R
            xlabels = '$f$'
            ylabels = '$R$'
        if control_seed == 0:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5, label='activation simulation')
        else:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5)

    def R_f_activation_diffthreshold_unstable(self, network_seed, control_seed, reverse):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect
        if betaeffect:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_diffthreshold_unstable/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}.csv'
        else:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_diffthreshold_unstable/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}.csv'

        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        f = data[:, 0]
        active_state = data[:, 1:]
        N_actual = np.size(active_state, 1)
        R = np.sum(active_state, 1) / N_actual
        if reverse == 'reverse':
            x = 1-f
            y = 1-R
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f
            y = R
            xlabels = '$f$'
            ylabels = '$R$'
        if control_seed == 0:
            #plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5, label='activation simulation')
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5, label='percolation')
        else:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5)

    def R_f_activation_iteratestate(self, network_seed, control_seed, reverse, ratio_threshold, arguments, iteration_number):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect, attractor_low, attractor_high = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect, self.attractor_low, self.attractor_high 
        dynamics_multi = globals()[dynamics + '_multi']
        dynamics_1D = globals()[dynamics + '_1D']
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
        beta_calculate = betaspace(A, [0])[0]
        w = np.sum(A, 0)
        N_actual = len(A)
        initial_low = np.ones(N_actual) * attractor_low
        initial_high = np.ones(N_actual) * attractor_high
        t = np.arange(0, 1000, 0.01)
        net_arguments = (index_i, index_j, A_interaction, cum_index)
        xL, xH = odeint(dynamics_1D, np.array([attractor_low, attractor_high]), np.arange(0, 1000, 0.01), args=(beta_calculate, arguments))[-1]
        xH_multi = odeint(dynamics_multi, initial_high, t, args=(arguments, net_arguments))[-1]
        xL_multi = odeint(dynamics_multi, initial_low, t, args=(arguments, net_arguments))[-1]
        xs_critical = ratio_threshold * xH_multi + (1 - ratio_threshold) * xL_multi

        if betaeffect:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_iteratestate/iteration_number={iteration_number}/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}.csv'
        else:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/percolation_activation_iteratestate/iteration_number={iteration_number}/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}.csv'

        data = np.array(pd.read_csv(des_file, header=None).iloc[:100, :])
        f = data[:, 0]
        state_xs = data[:, 1:]
        R = np.zeros(np.size(f))
        for i, f_i in enumerate(f):
            xs_select = state_xs[i]
            noncontrol_index = np.where(abs(xs_select-xH) > 1e-5)[0]
            R_i = np.sum(xs_select[noncontrol_index] >= xs_critical[noncontrol_index]) / N_actual
            R[i] = R_i + f_i
        if reverse == 'reverse':
            x = 1-f
            y = 1-R
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f
            y = R
            xlabels = '$f$'
            ylabels = '$R$'
        if control_seed == 0:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5, label='iteration simulation')
        else:
            plt.plot(x, y, '--', color='#e78ac3', linewidth=1.5)

    def R_f_dynamics(self, network_seed, control_seed, reverse, ratio_threshold, arguments):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect, attractor_low, attractor_high = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect, self.attractor_low, self.attractor_high 
        dynamics_decouple = globals()[dynamics + '_decouple']
        dynamics_1D = globals()[dynamics + '_1D']
        dynamics_multi = globals()[dynamics + '_multi']
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
        beta_calculate = betaspace(A, [0])[0]
        w = np.sum(A, 0)
        N_actual = len(A)
        initial_low = np.ones(N_actual) * attractor_low
        initial_high = np.ones(N_actual) * attractor_high
        t = np.arange(0, 1000, 0.01)
        xL, xH = odeint(dynamics_1D, np.array([attractor_low, attractor_high]), np.arange(0, 1000, 0.01), args=(beta_calculate, arguments))[-1]
        #xL_decouple = odeint(dynamics_decouple, initial_low, t, args=(xL, w, arguments))[-1]
        #xH_decouple = odeint(dynamics_decouple, initial_high, t, args=(xH, w, arguments))[-1]
        #xs_critical = ratio_threshold * xH_decouple + (1 - ratio_threshold) * xL_decouple

        net_arguments = (index_i, index_j, A_interaction, cum_index)
        xH_multi = odeint(dynamics_multi, initial_high, t, args=(arguments, net_arguments))[-1]
        xL_multi = odeint(dynamics_multi, initial_low, t, args=(arguments, net_arguments))[-1]
        xs_critical = ratio_threshold * xH_multi + (1 - ratio_threshold) * xL_multi

        #xs_critical = attractor_low
        if betaeffect:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/multi_dynamics/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'
            #save_file = '../report/report120721/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_beta={beta}_compare_f_R_diffthreshold={ratio_threshold}_' + reverse + '.png'
            save_file = '../report/report120721/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_beta={beta}_compare_f_R_dynamics_percolation_diffthreshold={ratio_threshold}_' + reverse + '.png'
        else:
            des_file =  '../data/' + dynamics + '/' + network_type + f'/multi_dynamics/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}_multi_xs.csv'
            #save_file = '../report/report120721/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_weight={beta}_compare_f_R_diffthreshold={ratio_threshold}_' + reverse + '.png'
            save_file = '../report/report120721/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_weight={beta}_compare_f_R_dynamics_percolation_diffthreshold={ratio_threshold}_' + reverse + '.png'

        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        f = data[:, 0]
        xs_multi = data[:, 1:]
        N_actual = np.size(xs_multi, 1)
        R = np.zeros(np.size(f))
        for i, f_i in enumerate(f):
            xs_select = xs_multi[i]
            noncontrol_index = np.where(abs(xs_select-xH) > 1e-5)[0]
            R_i = np.sum(xs_select[noncontrol_index] >= xs_critical[noncontrol_index]) / N_actual
            R[i] = R_i + f_i

        if reverse == 'reverse':
            x = 1-f
            y = 1-R
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f
            y = R
            xlabels = '$f$'
            ylabels = '$R$'
        if control_seed == 0:
            plt.plot(x, y, '--', color='#66c2a5', linewidth=1.5, label='dynamics')
        else:
            plt.plot(x, y, '--', color='#66c2a5', linewidth=1.5)

        plt.legend(fontsize=legendsize, frameon=False)
        
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
        plt.xlabel(xlabels, fontsize=fs)
        plt.ylabel(ylabels, fontsize=fs)
        plt.locator_params(axis="x", nbins=6)

        if control_seed == self.control_seed_list[-1]:
            #plt.savefig(save_file)
            #plt.close()
            pass

    def R_f_percolation(self, network_seed, reverse):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect, attractor_high = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect, self.attractor_high 
        """
        A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
        k = np.sum(A>0, 0)
        N_actual = len(k)
        if betaeffect:
            weight = A.max()
        else:
            weight = beta
            beta_cal = betaspace(A, [0])[0]

        n_H = np.array(nH_k_sL_sH(dynamics, beta_cal, weight))
        k_list = np.arange(0, len(n_H), 1)
        k_unrecovery = np.hstack(( k_list[n_H < 0], k_list[np.where(n_H > k_list)[0]]))
        num_unrecovery = sum([np.sum(k == i) for i in k_unrecovery])
        """


        percolation_dir = '../data/' + dynamics + '/' + network_type +  f'/percolation/'
        if betaeffect:
            des_file = percolation_dir + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_percolation.csv'
        else:
            des_file = percolation_dir + f'N={N}_d={d}_netseed={network_seed}_weight={beta}_percolation.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        f = data[:, 0]
        R = data[:, 1]
        sort_index = np.argsort(f)
        f_sort = f[sort_index]
        R_sort = R[sort_index]
        if reverse == 'reverse':
            x = 1-f_sort 
            y = 1-R_sort
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f_sort
            y = R_sort
            xlabels = '$f$'
            ylabels = '$R$'
        plt.plot(x, y, color='#fc8d62', linewidth=3, label='percolation theory')

    def R_f_percolation_diffstate(self, network_seed, reverse):
        """TODO: Docstring for R_f.

        :arg1: TODO
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect, attractor_high = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect, self.attractor_high 
        percolation_dir = '../data/' + dynamics + '/' + network_type +  f'/percolation_diffstate/'
        if betaeffect:
            des_file = percolation_dir + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_percolation.csv'
        else:
            des_file = percolation_dir + f'N={N}_d={d}_netseed={network_seed}_weight={beta}_percolation.csv'
        data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
        f = data[:, 0]
        R = data[:, 1]
        sort_index = np.argsort(f)
        f_sort = f[sort_index]
        R_sort = R[sort_index]
        if reverse == 'reverse':
            x = 1-f_sort 
            y = 1-R_sort
            xlabels = '$1-f$'
            ylabels = '$1-R$'
        else:
            x = f_sort
            y = R_sort
            xlabels = '$f$'
            ylabels = '$R$'
        plt.plot(x, y, color='#fc8d62', linewidth=3, label='percolation theory')

    def R_f_dynamics_average_control_seed(self, reverse):
        """TODO: Docstring for R_f_dynamics_average_control_seed.
        :returns: TODO

        """
        dynamics, network_type, N, d, beta, betaeffect, attractor_high, control_seed_list, network_seed_list = self.dynamics, self.network_type, self.N, self.d, self.beta, self.betaeffect, self.attractor_high, self.control_seed_list, self.network_seed_list
        y_list = []
        heterogeneity_list = []
        for network_seed in network_seed_list:
            y = []
            A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
            k = np.sum(A>0, 0)
            heterogeneity_list.append(np.round((np.mean(k**2) - np.mean(k)**2) / (np.mean(k)), 2))
            for control_seed in control_seed_list:
                if betaeffect:
                    des_file =  '../data/' + dynamics + '/' + network_type + f'/multi_dynamics/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'
                    save_file = '../report/report102621/' + network_type + f'_N={N}_d={d}_beta={beta}_compare_f_R_diffstate_avecontrolseed_' + reverse + '.png'
                else:
                    des_file =  '../data/' + dynamics + '/' + network_type + f'/multi_dynamics/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}_multi_xs.csv'
                    save_file = '../report/report102621/' + network_type + f'_N={N}_d={d}_weight={beta}_compare_f_R_diffstate_avecontrolseed_' + reverse + '.png'

                data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
                f = data[:, 0]
                xs_multi = data[:, 1:]
                N_actual = np.size(xs_multi, 1)
                R = np.sum(xs_multi > attractor_high, 1) / N_actual
                if reverse == 'reverse':
                    x = 1-f
                    y.append(1-R)
                    xlabels = '$1-f$'
                    ylabels = '$1-R$'
                else:
                    x = f
                    y.append(R)
                    xlabels = '$f$'
                    ylabels = '$R$'
            y_list.append(np.mean(np.array(y), 0))
        order = np.argsort(heterogeneity_list)
        for index in order:
            plt.plot(x, y_list[index], '--', linewidth=lw, label=f'h={heterogeneity_list[index]}')

        plt.legend(fontsize=legendsize*0.8, frameon=False, loc=1, bbox_to_anchor=(1.38,1.0) )
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.subplots_adjust(left=0.15, right=0.78, wspace=0.25, hspace=0.25, bottom=0.15, top=0.90)
        plt.xlabel(xlabels, fontsize=fs)
        plt.ylabel(ylabels, fontsize=fs)

        plt.savefig(save_file)
        plt.close('all')



    def data_reduction(self):
        des = self.des 
        network_type, N, weight, network_seed, d, control_constant, control_seed, method, interval = self.network_type, self.N, self.weight, self.network_seed, self.d, self.control_constant, self.control_seed, self.method, self.interval 
        des_reduction_file = des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}.csv'
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
        network_type, N, weight, network_seed, d, control_constant, control_seed, method, interval = self.network_type, self.N, self.weight, self.network_seed, self.d, self.control_constant, self.control_seed, self.method, self.interval
        des_multi_file = des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_multi_xs.csv'
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
        network_type, N, weight, network_seed, d, control_constant, control_seed, method, interval = self.network_type, self.N, self.weight, self.network_seed, self.d, self.control_constant, self.control_seed, self.method, self.interval
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
        save_des = '../report/report092821/' + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_' + method + '_'
        if method == 'KNN_connectcontrol':
            save_des += f'interval={interval}_'
        if plot_num == 'all':
            plt.savefig(save_des + 'xeff_comp.png')
        elif plot_num == 'major':
            plt.savefig(save_des + 'xeff_comp_major.png')
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

def plot_state_dis(network_type, N, beta, betaeffect, network_seed, d, control_seed, control_num_list, dynamics, arguments):
    """TODO: Docstring for plot_ratio_dis.

    :arg1: TODO
    :returns: TODO

    """
    if betaeffect:
        des_file =  '../data/' + dynamics + '/' + network_type + f'/multi_dynamics/N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'
        save_file = '../report/report110921/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_beta={beta}_compare_f_R_diffthreshold={ratio_threshold}_' + reverse + '.png'
    else:
        des_file =  '../data/' + dynamics + '/' + network_type + f'/multi_dynamics/N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}_multi_xs.csv'
        save_file = '../report/report110921/' + network_type + f'_N={N}_d={d}_netseed={network_seed}_weight={beta}_compare_f_R_diffthreshold={ratio_threshold}_' + reverse + '.png'
    dynamics_decouple = globals()[dynamics + '_decouple']
    dynamics_1D = globals()[dynamics + '_1D']
    dynamics_multi = globals()[dynamics + '_multi']
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    beta_calculate = betaspace(A, [0])[0]
    w = np.sum(A, 0)
    N_actual = len(A)
    initial_low = np.ones(N_actual) * attractor_low
    initial_high = np.ones(N_actual) * attractor_high
    t = np.arange(0, 1000, 0.01)
    xL, xH = odeint(dynamics_1D, np.array([attractor_low, attractor_high]), t, args=(beta_calculate, arguments))[-1]
    xL_decouple = odeint(dynamics_decouple, initial_low, t, args=(xL, w, arguments))[-1]
    xH_decouple = odeint(dynamics_decouple, initial_high, t, args=(xH, w, arguments))[-1]
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    xH_multi = odeint(dynamics_multi, initial_high, t, args=(arguments, net_arguments))[-1]
    xL_multi = odeint(dynamics_multi, initial_low, t, args=(arguments, net_arguments))[-1]

    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    f = data[:, 0]
    xs_multi = data[:, 1:]
    """
    delta_list = []
    for f_i in f:
        index = np.where(abs(f_i-f)<1e-8)[0][0]
        xs_select = xs_multi[index]
        noncontrol_index = np.where(abs(xs_select-xH) > 1e-5)[0]
        delta_list.append( np.mean(((xs_select - xL_multi) / (xH_multi - xL_multi))[noncontrol_index]) )
    plt.plot(f, delta_list, '.')
    plt.show()
    """
    for control_num in control_num_list:
        index = np.where(abs(f-control_num)<1e-8)[0][0]
        xs_select = xs_multi[index]
        noncontrol_index = np.where(abs(xs_select-xH) > 1e-5)[0]
        data_analysis = ((xs_select - xL_multi) / (xH_multi - xL_multi))[noncontrol_index]

        #data_analysis = xs_select[noncontrol_index]
        noncontrol_size = np.size(noncontrol_index)
        #state_distribution, x = np.histogram(xs_select, bins=np.linspace(xs_select.min(), xs_select.max() * 1.01, 100))
        bins = np.logspace(np.log10(data_analysis.min()), np.log10(data_analysis.max()), 20)
        state_distribution, x = np.histogram(data_analysis, bins=bins)
        bin_centers = 0.5*(x[1:]+x[:-1])
        plt.semilogx(bin_centers, state_distribution / sum(state_distribution) * N_actual, marker='o', linewidth=lw, linestyle = '-', label=f'$f=${control_num}')
    plt.legend(frameon=False, fontsize = legendsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$\\delta$', fontsize=fs)
    plt.ylabel('distribution', fontsize=fs)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    #plt.show()
    return None

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

def compare_state_multi_decouple(network_type, N, d, beta, betaeffect, network_seed, dynamics, arguments, attractor_value):
    """TODO: Docstring for compare_state_multi_decouple.

    :network_type: TODO
    :N: TODO
    :d: TODO
    :network_seed: TODO
    :dynamics: TODO
    :arguments: TODO
    :: TODO
    :returns: TODO

    """
    dynamics_decouple = globals()[dynamics + '_decouple']
    dynamics_1D = globals()[dynamics + '_1D']
    dynamics_multi = globals()[dynamics + '_multi']
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, beta, betaeffect, network_seed, d)
    beta_calculate = betaspace(A, [0])[0]
    w = np.sum(A, 0)
    N_actual = len(A)
    initial_condition = np.ones(N_actual) * attractor_value
    t = np.arange(0, 1000, 0.01)
    net_arguments = (index_i, index_j, A_interaction, cum_index)
    xs_multi = odeint(dynamics_multi, initial_condition, t, args=(arguments, net_arguments))[-1]
    x_eff = odeint(dynamics_1D, initial_condition[0], t, args=(beta_calculate, arguments))[-1]
    xs_decouple = odeint(dynamics_decouple, initial_condition, t, args=(x_eff, w, arguments))[-1]
    plt.plot(xs_multi, xs_decouple, '.')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.locator_params(nbins=6)
    plt.xlabel('$x_s^{multi}$', fontsize=fs)
    plt.ylabel('$x_s^{decouple}$', fontsize=fs)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.show()

    return xs_multi, x_eff, xs_decouple


def plot_fixedpoint_controlsize(network_type, dynamics, network_seed, N, d, beta, betaeffect, control_seed):
    """TODO: Docstring for R_f.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/fixedpoint_controlsize/'  
    if betaeffect == 0:
        des_file = des + f'N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}_multi_xs.csv'
    else:
        des_file = des + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    control_constant = Counter(np.round(data[-1], 12)).most_common(1)[0][0]

    f = data[::3, 0]
    xs_multi_low = data[::3, 1:]
    xs_multi_unstable = data[1:][::3, 1:]
    xs_multi_high = data[2:][::3, 1:]
    N_actual = np.size(xs_multi_low, 1)
    plt.plot(f, (np.sum(xs_multi_unstable, 1) - control_constant  * f * N_actual) / (N_actual * ( 1-f)), '--',) 
    plt.plot(f, (np.sum(xs_multi_low, 1) - control_constant  * f * N_actual) / (N_actual * ( 1-f)), linewidth=lw, alpha=alpha) 
    plt.plot(f, (np.sum(xs_multi_high, 1) - control_constant  * f * N_actual) / (N_actual * ( 1-f)), linewidth=lw, alpha=alpha ) 

    plt.legend(fontsize=legendsize, frameon=False)
    
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xlabel('$f$', fontsize=fs)
    plt.ylabel('$\\langle x \\rangle$', fontsize=fs)
    #plt.locator_params(axis="x", nbins=6)
    plt.show()

def plot_unstable_controlsize(network_type, dynamics, network_seed, N, d, beta, betaeffect, control_seed, y_f):
    """TODO: Docstring for R_f.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/fixedpoint_controlsize/'  
    if betaeffect == 0:
        des_file = des + f'N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}_multi_xs.csv'
    else:
        des_file = des + f'N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'
    data = np.array(pd.read_csv(des_file, header=None).iloc[:, :])
    control_constant = Counter(np.round(data[-1], 12)).most_common(1)[0][0]

    f = data[::3, 0]
    xs_multi_low = data[::3, 1:]
    xs_multi_unstable = data[1:][::3, 1:]
    xs_multi_high = data[2:][::3, 1:]
    N_actual = np.size(xs_multi_low, 1)
    index_y_f = np.where(np.abs(f-y_f)<1e-10)[0][0]
    index_uncontrol = np.where(np.abs(xs_multi_unstable[index_y_f] - control_constant)  >1e-5)[0]
    xs_unstable_uncontrol = xs_multi_unstable[:, index_uncontrol]

    #plt.plot(xs_unstable_uncontrol[0], xs_unstable_uncontrol[index_y_f], '.') 
    plt.plot(xs_unstable_uncontrol[0], xs_multi_low[index_y_f, index_uncontrol], '.') 
    plt.plot(np.linspace(0, np.max(np.hstack((xs_unstable_uncontrol[0], xs_unstable_uncontrol[index_y_f])))  * 1.1, 10), np.linspace(0, np.max(np.hstack((xs_unstable_uncontrol[0], xs_unstable_uncontrol[index_y_f])))  * 1.1, 10), '--', linewidth=lw, alpha=alpha) 

    plt.legend(fontsize=legendsize, frameon=False)
    
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xlabel('$x^u (f=0)$', fontsize=fs)
    #plt.ylabel(f'$x^u (f={y_f}$)', fontsize=fs)
    plt.ylabel(f'$x^L (f={y_f}$)', fontsize=fs)
    #plt.locator_params(axis="x", nbins=6)
    plt.show()





dynamics = 'mutual'
arguments = (B, C, D, E, H, K)
threshold = 5
attractor_low = 0.1
attractor_high = 5

dynamics = 'genereg'
arguments = (B_gene, )
attractor_high = 10
attractor_low = 0.1


N = 1000
beta = 1
control_seed_list = [0]
control_seed_list = np.arange(0, 10, 1).tolist() 

control_num = 0.2



betaeffect = 0











network_type = 'SF'
beta = 0.18
d = [3, 0, 4]
d = [3.8, 0, 5]
d = [2.5, 0, 3]
d_list = [[2.1, 0, 2], [2.1, 0, 3], [2.5, 0, 2], [2.5, 0, 3], [3, 0, 4], [3, 0, 3], [3, 0, 2], [3.8, 0, 4], [3.8, 0, 3], [3.8, 0, 2], [3.8, 0, 5]]
d_list = [[2.5, 0, 3]]
network_seed_list = np.tile(np.arange(0, 1, 1), (2, 1)).transpose().tolist()
network_seed = [0, 0]
ratio_threshold_list = [0.25]

network_type = 'ER'
beta = 0.13
d = 4000
d_list = [8000]
network_seed_list = np.arange(0, 1, 1).tolist()
network_seed = 0
ratio_threshold_list = [0.9]



reverse = 'reverse'
reverse = 'nonreverse'
ratio_threshold = 0.15
iteration_number = 30

for d in d_list:
    data = data_plot(network_type, N, beta, betaeffect, network_seed_list, d, control_seed_list, control_num, dynamics, attractor_high, attractor_low)
    #data.R_f_dynamics_average_control_seed(reverse)

    for network_seed in network_seed_list:
        #data.R_f_percolation(network_seed, reverse)
        #data.R_f_percolation_diffstate(network_seed, reverse)
        for ratio_threshold in ratio_threshold_list: 
            for control_seed in control_seed_list:
                #data.R_f_activation(network_seed, control_seed, reverse)
                #data.R_f_activation_diffstate(network_seed, control_seed, reverse)
                #data.R_f_activation_diffthreshold(network_seed, control_seed, reverse, ratio_threshold)
                #data.R_f_activation_iteratestate(network_seed, control_seed, reverse, ratio_threshold, arguments, iteration_number)
                #data.R_f_dynamics(network_seed, control_seed, reverse, ratio_threshold, arguments)
                #data.R_f_activation_diffthreshold_unstable(network_seed, control_seed, reverse)
                pass


#data.plot_sH_sL_multi()
#data.threshold_line(beta)
control_num_list =[0.4]
#plot_state_dis(network_type, N, beta, betaeffect, network_seed, d, control_seed, control_num_list, dynamics, arguments)

attractor_value = attractor_high
#compare_state_multi_decouple(network_type, N, d, beta, betaeffect, network_seed, dynamics, arguments, attractor_value)

control_seed = 0
#plot_fixedpoint_controlsize(network_type, dynamics, network_seed, N, d, beta, betaeffect, control_seed)
y_f = 0.06
plot_unstable_controlsize(network_type, dynamics, network_seed, N, d, beta, betaeffect, control_seed, y_f)
