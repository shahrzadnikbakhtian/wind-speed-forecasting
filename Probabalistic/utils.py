import math
import os
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import sys

sys.path.append('c:/Users/jerry/Desktop/oslomet2023/Thesis/p6/generative-models')


import dataset



#test_size:int=2*12*2
def load_data(data, random_state: int = 0, test_size:int=100):
   
    df_load = data

    features = ['elev_meps_0', 'z_l500_0', 't2_0', 't_l850_0', 'rh2_0', 'ws10_0', 'gust10_0','tcc_0', 'mslp_0']
    max_load = df_load['ws_obs']
    max_load_z=df_load['ws10_0']

    nb_days = int(len(df_load) / 6)
    x = np.concatenate([df_load[col].values.reshape(nb_days, 6) for col in features], axis=1)
    y = df_load['ws_obs'].values.reshape(nb_days, 6)
    w = df_load['ws10_0'].values.reshape(nb_days, 6) 
    df_x = pd.DataFrame(data=x)
    df_w = pd.DataFrame(data=w)

    #df_y = pd.DataFrame(data=y, index=df_load.index)
    #df_x = pd.DataFrame(data=x, index=df_load.index) 

    # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
    df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,
                                                                    random_state=random_state, shuffle=True)
    df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,
                                                          random_state=random_state, shuffle=True)

    nb_days_LS = len(df_y_LS)
    nb_days_VS = len(df_y_VS)
    nb_days_TEST = len(df_y_TEST)
    print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

    return df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST, df_w



# def wind_data(data, random_state=0, test_size=2*12*2):
    
#     df_wind= data
#     ZONES = ['ZONE_' + str(i) for i in range(1, 2)]
#     features = ['elev_meps_0', 'z_l500_0', 't2_0', 't_l850_0', 'rh2_0', 'gust10_0','tcc_0', 'mslp_0', 'ws10_0', 'ws_l850_0']
#     max_wind = df_wind['ws_obs'].max()
    
#     data_zone = []
#     for zone in ZONES:
#         df_var = df_wind[df_wind[zone] == 1].copy()
#         nb_days = int(len(df_var) / 24)
#         zones = [df_var[zone].values.reshape(nb_days, 24)[:, 0].reshape(nb_days, 1) for zone in ZONES]
#         x = np.concatenate([df_var[col].values.reshape(nb_days, 24) for col in features] + zones, axis=1)
#         y = df_var['ws_obs'].values.reshape(nb_days, 24) /max_wind
#         df_y = pd.DataFrame(data=y, index=df_var['ws_obs'].asfreq('D').index)
#         df_x = pd.DataFrame(data=x, index=df_var['ws_obs'].asfreq('D').index)

#         # Decomposition between LS, VS & TEST sets (TRAIN = LS + VS)
#         df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,random_state=random_state, shuffle=True)
#         df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,random_state=random_state, shuffle=True)

#         data_zone.append([df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST])

#         nb_days_LS = len(df_y_LS)
#         nb_days_VS = len(df_y_VS)
#         nb_days_TEST = len(df_y_TEST)
#         print('#LS %s days #VS %s days # TEST %s days' % (nb_days_LS, nb_days_VS, nb_days_TEST))

#     return [pd.concat([data_zone[i][j] for i in range(0, 1)], axis=0, join='inner') for j in range(0, 5 + 1)]


def dump_file(dir:str, name: str, file):
    """
    Dump a file into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'wb')
    pickle.dump(file, file_name)
    file_name.close()

def read_file(dir:str, name: str):
    """
    Read a file dumped into a pickle.
    """
    file_name = open(dir + name + '.pickle', 'rb')
    file = pickle.load(file_name)
    file_name.close()

    return file

if __name__ == "__main__":
    # Set the working directory to the root of the project
    print(os.getcwd())
    path = 'C:/Users/jerry/Desktop/oslomet2023/Thesis/meps+obs_mbr000_00.rds'
    #path = 'meps+obs_mbr000_00.rds'
    # --------------------------------------------------------------------------------------------------------------
    # NEW DATASETS
    # --------------------------------------------------------------------------------------------------------------
    
    load_data = load_data(dataset.load_dataset(path), test_size=100, random_state=0)

