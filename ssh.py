""" transfer files between local machine and remote server"""
import paramiko
import os 
import numpy as np 

client = paramiko.SSHClient()
client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect('ganxis3.nest.rpi.edu', username='mac6', password='woods*score&sister')

def transfer_files(directory, filenames):

    server_des = '/home/mac6/RPI/research/nonlinear_control/data/' 
    local_des = '/home/mac/RPI/research/nonlinear_control/data/'
    if not os.path.exists(local_des):
        os.makedirs(local_des)
    sftp = client.open_sftp()
    if '/' in directory:
        if not os.path.exists(local_des + directory):
            os.makedirs(local_des + directory)
        filenames = sftp.listdir(server_des+directory) 
    for i in filenames:
        sftp.get(server_des + directory + i, local_des + directory +i)
    sftp.close()

dynamics = 'mutual_multi'
network_type = '2D'
network_type = 'ER'
beta = 1
network_seed = 0
beta_list = np.round(np.arange(0.4, 1.4, 0.02), 2)
beta_list = [1]
beta_list = [0.2]
betaeffect=0
N_list = [900]
d = 7200
control_num_list = [5, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90]
control_num_list = np.arange(200, 300, 100).astype(int).tolist()
control_num_list = [50, 100, 200]
control_num_list = [0.041, 0.042]
N_list = [10000]
N = 10000
d_list = np.array(np.array([4]) * N/2, dtype=int)
beta_list = [0.2]





def transfer_dynamics(network_type, dynamics, d_list, beta_list, network_seed_list, N_list, control_num_list, control_value_list):
    """TODO: Docstring for transfer_dynamics.

    :network_type: TODO
    :dynamics: TODO
    :d_list: TODO
    :beta_list: TODO
    :network: TODO
    :returns: TODO

    """
    for d in d_list:
        for beta in beta_list:
            ratio_dir = '../data/' +  dynamics + '/' + network_type + f'/c={d}/'
            if betaeffect:
                ratio_dir = ratio_dir + f'/beta={beta}/'
            else:
                ratio_dir = ratio_dir + f'/edgewt={beta}/'


            if not os.path.exists(ratio_dir):
                os.makedirs(ratio_dir)
            for network_seed in network_seed_list:
                for N in N_list:
                    for control_num in control_num_list:
                        for control_value in control_value_list:
                            filenames = ratio_dir +  f'netseed={network_seed}_N={N}_control_num={control_num}_value={control_value}.csv'

                            transfer_files('', [filenames])

def transfer_percolation(network_type, c_list, k_list, network_seed_list, N_list, f_list):
    """TODO: Docstring for transfer_percolation.

    :network_type: TODO
    :c_list: TODO
    :k_list: TODO
    :network: TODO
    :returns: TODO

    """
    for c in c_list:
        for k in k_list:
            directory = '../data/percolation/' + network_type + f'/c={c}/k={k}/'
            #transfer_files('', ['theory.csv'])
            if not os.path.exists(directory):
                os.makedirs(directory)
            for network_seed in network_seed_list:
                for N in N_list:
                    for f in f_list:
                        filenames = directory +  f'netseed={network_seed}_N={N}_f={f}.csv'
                        transfer_files('', [filenames])

def transfer_dynamics_reduction(network_type, N, weight_list, network_seed, d_list, dynamics, control_constant, control_seed_list, method, interval):
    """TODO: Docstring for transfer_dynamics.

    :network_type: TODO
    :dynamics: TODO
    :d_list: TODO
    :beta_list: TODO
    :network: TODO
    :returns: TODO

    """
    if method == 'KNN':
        des = '../data/' + dynamics + '/' + network_type + f'/KNN/'
    elif method == 'firstNN':
        des = '../data/' + dynamics + '/' + network_type + f'/firstNN/'
    elif method == 'KNN_connectcontrol':
        des = '../data/' + dynamics + '/' + network_type + f'/KNN_subgroup_interval={interval}/'

    if not os.path.exists(des):
        os.makedirs(des)
    for d in d_list:
        for weight in weight_list:
            for control_seed in control_seed_list:
                des_reduction_file = des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}.csv'
                des_multi_file = des + f'N={N}_d={d}_netseed={network_seed}_wt={weight}_controlseed={control_seed}_controlconstant={control_constant}_multi_xs.csv'
                transfer_files('', [des_reduction_file, des_multi_file])
    return None

def transfer_dynamics_recovery(network_type, N, d, network_seed_list, dynamics, beta_list, betaeffect, control_seed_list):
    """TODO: Docstring for transfer_dynamics.

    :network_type: TODO
    :dynamics: TODO
    :d_list: TODO
    :beta_list: TODO
    :network: TODO
    :returns: TODO

    """
    ratio_dir = '../data/' +  dynamics + '/' + network_type + '/multi_dynamics/'
    if not os.path.exists(ratio_dir):
        os.makedirs(ratio_dir)
    for network_seed in network_seed_list:
        for control_seed in control_seed_list:
            for beta in beta_list:
                if betaeffect:
                    filenames = ratio_dir +  f'N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}_multi_xs.csv'
                else:
                    filenames = ratio_dir +  f'N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}_multi_xs.csv'

                transfer_files('', [filenames])

def transfer_percolation_activation(network_type, N, d, network_seed_list, dynamics, beta_list, betaeffect, control_seed_list):
    """TODO: Docstring for transfer_dynamics.

    :network_type: TODO
    :dynamics: TODO
    :d_list: TODO
    :beta_list: TODO
    :network: TODO
    :returns: TODO

    """
    ratio_dir = '../data/' +  dynamics + '/' + network_type + '/percolation_activation_diffstate/'
    if not os.path.exists(ratio_dir):
        os.makedirs(ratio_dir)
    for network_seed in network_seed_list:
        for control_seed in control_seed_list:
            for beta in beta_list:
                if betaeffect:
                    filenames = ratio_dir +  f'N={N}_d={d}_netseed={network_seed}_beta={beta}_controlseed={control_seed}.csv'
                else:
                    filenames = ratio_dir +  f'N={N}_d={d}_netseed={network_seed}_weight={beta}_controlseed={control_seed}.csv'

                transfer_files('', [filenames])
    return None







network_type = 'ER'
c_list = [16]
k_list = [4]
network_seed_list = [0]
N_list = [1000, 10000, 100000, 1000000]
N_list = [1000000]
f_list = np.round(np.arange(0.0042, 0.0051, 0.0002), 4)
f_list = np.round(np.arange(0.055, 0.08, 0.005), 3)
f_list = np.round(np.arange(0.041, 0.051, 0.002), 3)
f_list = [0.03, 0.04, 0.05, 0.06, 0.07]

k_list = [7]
N_list = [1000000]
f_list = np.round(np.arange(0.16, 0.170001, 0.001), 3) 
f_list = [0.1, 0.12, 0.14, 0.18, 0.2]


#transfer_percolation(network_type, c_list, k_list, network_seed_list, N_list, f_list)

d_list = [16]
beta_list = [0.1]
network_seed_list = [0]
N_list = [10000]
control_num_list =  np.round(np.arange(0.01, 0.8, 0.001), 3)
control_value_list = [1]

beta_list = [0.05]
N_list = [10000]
control_num_list = np.round(np.arange(0.16, 0.18, 0.001), 3)

dynamics = 'mutual'
network_type = 'ER'
N = 1000
weight_list = [0.1]
network_seed = 0
d_list = [8000]
control_constant = 5
control_seed_list = np.arange(100).tolist()
method_list = ['KNN', 'firstNN', 'KNN_connectcontrol']
interval = 1
for method in method_list:
    #transfer_dynamics_reduction(network_type, N, weight_list, network_seed, d_list, dynamics, control_constant, control_seed_list, method, interval)
    pass

dynamics = 'mutual'
N = 1000
betaeffect = 0
beta_list = [0.1]
control_seed_list = np.arange(10).tolist()



network_type = 'SF'
d = [2.5, 0, 3]
d_list = [[2.1, 0, 2], [2.1, 0, 3], [2.5, 0, 2], [2.5, 0, 4], [3, 0, 3], [3, 0, 2], [3.8, 0, 4], [3.8, 0, 3], [3.8, 0, 2]]
d_list = [[2.1, 0, 3]]
network_seed_list = np.tile(np.arange(0, 10, 1), (2, 1)).transpose().tolist()


network_type = 'ER'
d = 3000
d_list = [2000, 4000]
network_seed_list = np.arange(0, 10, 1).tolist()




for d in d_list:
    #transfer_dynamics_recovery(network_type, N, d, network_seed_list, dynamics, beta_list, betaeffect, control_seed_list)
    transfer_percolation_activation(network_type, N, d, network_seed_list, dynamics, beta_list, betaeffect, control_seed_list)
