import numpy as np
from numpy.testing import *

from src.network.network_model import NetworkModel
from src.network.sumo_network import SUMONetwork
from src.prediction_models.neural_nets.global_fcn import GlobalFCNPredictor
from src.prediction_models.neural_nets.nn_predictor import NNPredictor
from src.traffic_data.csv_traffic_data import CSVTrafficData
from src.traffic_data.traffic_data import TrafficData


def test_get_network_data():
    net = SUMONetwork('tests/test_cases/network_1.xml')
    data = CSVTrafficData('tests/test_cases/csv_data_2.csv', net)
    data.split()
    predictor = NNPredictor(data, 0, 0, 0, 0, ['speed'])
    predictor.seq_len = 1
    train, label = predictor.get_network_data('train')
    train = np.reshape(train, (train.shape[0], -1))
    label = np.reshape(label, (label.shape[0], -1))
    assert_allclose(train, [[50.2, 30.8, 20, 52]])
    assert_allclose(label, [[45.8, 52.6]])


def test_get_network_data2():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    assert len(data.links) == 56
    assert data.timestamps == [str(i) for i in range(1, 49)]
    data.split(val_pct=0.2)
    assert data.train_data.shape == (5, 56, 39)
    assert data.val_data.shape == (5, 56, 9)

    predictor = NNPredictor(data, 0, 0, 0, 0, target_feature=['count', 'flow', 'travel_time', 'speed', 'density'])
    predictor.seq_len = 1
    train, label = predictor.get_network_data('train')
    train = np.reshape(train, (train.shape[0], -1))
    label = np.reshape(label, (label.shape[0], -1))
    assert_allclose(train, data.data[:, :, :38].reshape((-1, 38)).T)
    assert_allclose(label, data.data[:, :, 1:39].reshape((-1, 38)).T)


# test past time steps
def test_get_network_data3():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    data.split(val_pct=0.2)
    predictor = GlobalFCNPredictor(data, 8, 8, past_steps=1)
    train, label = predictor.get_network_data('train')
    assert_allclose(train[:, 1, :], data.data[:, :, 1:38].reshape((-1, 37)).T)
    assert_allclose(train[:, 0, :], data.data[:, :, :37].reshape((-1, 37)).T)
    assert_allclose(label[:, 0, :], data.data[data.features.index('speed'), :, 2:39].T)


def test_remove_invalid_data():
    net = SUMONetwork('tests/test_cases/network_1.xml')
    test_data = TrafficData(net, 10)
    test_data.features = ['speed']
    test_data.links = net.link_ids
    test_data.timestamps = list(range(100))
    test_data.data = np.array([[range(1000, 1100), range(2000, 2100)]])
    test_data.split(test_pct=0.1)
    predictor = GlobalFCNPredictor(test_data, 8, 8, past_steps=2)
    data, label = predictor.get_network_data('train')
    data, label = predictor.remove_invalid_data(data, label)
    data = np.mod(data, 10)
    label = np.repeat(np.mod(label, 10), 3, axis=1)
    assert_array_less(data, label)
