import pickle

def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)

def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)