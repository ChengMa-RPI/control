import numpy as np
import pandas as pd

dynamics = 'mutual'
network_type = 'SF'
N = 1000
weight = 0.1
d_list = [[2.1, 0, 2], [2.1, 0, 3], [2.5, 0, 2], [2.5, 0, 4], [3, 0, 3], [3, 0, 2], [3.8, 0, 4], [3.8, 0, 3], [3.8, 0, 2]]
network_seed_list =  np.tile(np.arange(7, 8, 1), (2, 1)).transpose().tolist()
control_seed_list = np.arange(10).tolist()

 
for network_seed in network_seed_list:
    for d in d_list:
        for control_seed in control_seed_list:
            des_file = '../data/' + dynamics + '/' + network_type + '/percolation_activation_diffstate/' + f'N={N}_d={d}_netseed={network_seed}_weight={weight}_controlseed={control_seed}.csv'
            data = pd.read_csv(des_file, header=None)
            index = [np.where(np.abs(np.array(data)[:, 0] - i * 0.01) < 1e-8)[0][0] for i in range(100)]
            df = data.iloc[index]
            df.to_csv(des_file, index=False, header=False)


