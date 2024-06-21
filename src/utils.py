import pickle
import numpy as np
import pandas as pd

def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)

def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)

def load_electricity_dataset(data_dir):
    # ELEC2 data set
    # downloaded from https://www.kaggle.com/yashsharan/the-elec2-dataset
    data = pd.read_csv(data_dir)
    col_names = data.columns
    data = data.to_numpy()

    # remove the first stretch of time where 'transfer' does not vary
    data = data[17760:]

    # set up variables for the task (predicting 'transfer')
    covariate_col = ['nswprice', 'nswdemand', 'vicprice', 'vicdemand']
    response_col = 'transfer'
    # keep data points for 9:00am - 12:00pm
    keep_rows = np.where((data[:, 2] > data[17, 2]) & (data[:, 2] < data[24, 2]))[0]

    X = data[keep_rows][:, np.where([t in covariate_col for t in col_names])[0]]
    Y = data[keep_rows][:, np.where(col_names == response_col)[0]].flatten()
    X = X.astype('float64')
    Y = Y.astype('float64')

    return X, Y