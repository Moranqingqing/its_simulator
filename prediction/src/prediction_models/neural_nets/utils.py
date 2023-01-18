import csv

import numpy as np
import xarray

from src.network.pems_network import PeMSNetwork
from src.traffic_data.traffic_data import TrafficData


def convert_pems_to_traffic_data(path_pems, pems_csv):
    data = load_pems_data(path_pems)
    timestamps = list(range(data.shape[-1]))

    traffic_data = TrafficData()
    traffic_data.data = data
    traffic_data.timestamps = timestamps
    traffic_data.features = ['flow', 'density', 'speed']

    num_nodes = data.shape[1]
    traffic_data.links = [i for i in range(num_nodes)]

    traffic_data.data = traffic_data.data.assign_coords(coords={'feature': traffic_data.features, 'link': traffic_data.links, 'time': timestamps})

    traffic_data.network = PeMSNetwork(pems_csv, num_nodes)

    adjacency_matrix_connect = build_adj_matrix_pems(pems_csv, num_nodes, graph_type='connect')
    adjacency_matrix_distance = build_adj_matrix_pems(pems_csv, num_nodes, graph_type='distance')
    traffic_data.adjacency_matrix = {'connect': adjacency_matrix_connect,
                                     'distance': adjacency_matrix_distance}

    return traffic_data


def load_pems_data(path):
    """

    Parameters
    ----------
    path: str
    path to the npz file.

    Returns
    -------
    xarray shaped (F, N, T)
    """
    data = np.load(path)['data'].transpose(2, 1, 0)  # (T,N,F) -> (F,N,T)
    data = xarray.DataArray(data, dims=['feature', 'link', 'time'])
    return data


def build_adj_matrix_pems(pems_csv, num_nodes, graph_type='connect'):
    """

    Parameters
    ----------
    pems_csv : str
    path to the distance csv file

    num_nodes : int
    number of nodes

    graph_type : str
    'connect': Aij = 1 if connected;
    'distance': Aij = 1/d if connected.

    Returns
    -------
    numpy array
    """
    A = np.zeros((num_nodes, num_nodes))
    with open(pems_csv, "r") as f_d:
        f_d.readline()  # skip table head
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type should be either 'connect' or 'distance'")

    return A
