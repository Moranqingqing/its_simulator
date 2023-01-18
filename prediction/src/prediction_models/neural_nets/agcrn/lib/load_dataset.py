import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PeMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'qew-flow':
        data_path = os.path.join('../data/50runs-1min-flow.npz')
        data = np.load(data_path)['data']  # traffic flow data
    elif dataset == 'qew-speed':
        data_path = os.path.join('../data/50runs-1min-speed.npz')
        data = np.load(data_path)['data']  # traffic flow data
    elif dataset == 'urban-speed':
        data_path = os.path.join('../data/50runs-1min-urban-speed.npz')
        data = np.load(data_path)['data']  # traffic flow data
    else:
        raise ValueError
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
