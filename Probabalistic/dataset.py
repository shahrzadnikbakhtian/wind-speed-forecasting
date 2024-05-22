import pandas as pd
import numpy as np
import pyreadr

def load_dataset(data_path):
    # Create a sample timeseries dataset
    data_path= 'C:/Users/jerry/Desktop/oslomet2023/Thesis/meps+obs_mbr000_00.rds'
    index = pd.date_range(start='2021-11-01 00:00:00', end='2023-11-01 23:00:00', freq='H')
    series = pd.Series([x for x in range (1, 17545)], index=index)
    sample_df = pd.DataFrame({'s': series})
    #sample_df.asfreq(freq='D').index

    # Load dataset
    path = data_path
    dataset = pyreadr.read_r(path)
    loaded_df = dataset[None]
    
    # Convert time to datetime
    loaded_df['time'] = pd.to_datetime(loaded_df['time'])
    
    # Added zone for the target station
    #loaded_df['ZONE_1'] = 1
    target_station = 'SN10380'
    rows = loaded_df[loaded_df['sid'] == target_station]

    # Handling missing data
    df_interpolated = rows.interpolate(method='linear', axis=0)

    # Handling duplicated time
    df = df_interpolated.drop_duplicates(subset='time', keep='first')

    # Selected featues for dataset
    XY_selected = df[['time','ws_obs', 'elev_meps_0','ws10_0', 'z_l500_0', 't2_0', 't_l850_0', 'rh2_0', 'gust10_0','tcc_0', 'mslp_0']]
    XY_selected.drop(XY_selected.tail(1).index, inplace=True) # Drop the last line?

    # Time as index
    XY_selected.set_index('time', inplace=True)

    # Align the original dataset with the sample dataset to fill missing dates 
    sample_df,XY_selected = sample_df.align(XY_selected)

    # Handling Null data
    #missing_count = XY_selected.isnull().sum()
    #XY_selected['ws_obs'].asfreq('D')
    data = XY_selected.interpolate(method='linear', axis=0)
    data.drop(columns=['s'], inplace=True)

    return data