import numpy as np
from numpy.testing import *

from src.network.network_model import NetworkModel
from src.network.sumo_network import SUMONetwork
from src.prediction_models.regression import RegressionPredictor
from src.traffic_data.csv_traffic_data import CSVTrafficData
from src.traffic_data.traffic_data import TrafficData


def test_neighbors():
    net = SUMONetwork('tests/test_cases/network_1.xml')
    data = CSVTrafficData('tests/test_cases/csv_data_2.csv', net)
    data.split()
    predictor = RegressionPredictor(data, 'ridge')
    train, label = predictor.get_neighborhood_data('train', 'A')
    assert_allclose(train, [[50.2, 20]])
    assert_allclose(label, [[45.8]])


# test one location
def test_neighbors2():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    assert len(data.links) == 56
    assert data.timestamps == [str(i) for i in range(1, 49)]
    data.split(val_pct=0.2)
    assert data.train_data.shape == (5, 56, 39)
    assert data.val_data.shape == (5, 56, 9)

    predictor = RegressionPredictor(data, 'ridge', target_feature=['count', 'flow', 'travel_time', 'speed', 'density'])
    train, label = predictor.get_neighborhood_data('train', '4478')
    assert_allclose(train, data.data[:, data.links.index('4478'), :38].T)
    assert_allclose(label, data.data[:, data.links.index('4478'), 1:39].T)


# test location and 1 hop neighbor, no past time steps
def test_neighbors3():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    data.split(val_pct=0.2)

    predictor = RegressionPredictor(data, 'ridge', n_hops=1)
    train, label = predictor.get_neighborhood_data('train', '4478')
    assert_allclose(train[:, :5], data.data[:, data.links.index('4478'), :38].T)
    assert_allclose(train[:, 5::2], data.data[:, data.links.index('1321'), :38].T)
    assert_allclose(train[:, 6::2], data.data[:, data.links.index('5903'), :38].T)
    assert_allclose(label, data.data[data.features.index('speed'), data.links.index('4478'), 1:39].reshape(-1, 1))


# test location and 1 hop neighbor, including past time steps
def test_neighbors4():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    data.split(val_pct=0.2)
    predictor = RegressionPredictor(data, 'ridge', n_hops=1, past_steps=1)
    train, label = predictor.get_neighborhood_data('train', '4478')
    assert_allclose(train[:, :5], data.data[:, data.links.index('4478'), :37].T)
    assert_allclose(train[:, 15:20], data.data[:, data.links.index('4478'), 1:38].T)
    assert_allclose(label, data.data[data.features.index('speed'), data.links.index('4478'), 2:39].reshape(-1, 1))
    assert_allclose(train[:, 5:15:2], data.data[:, data.links.index('1321'), :37].T)
    assert_allclose(train[:, 6:16:2], data.data[:, data.links.index('5903'), :37].T)
    assert_allclose(train[:, 20:30:2], data.data[:, data.links.index('1321'), 1:38].T)
    assert_allclose(train[:, 21:31:2], data.data[:, data.links.index('5903'), 1:38].T)


def test_model_creation():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    data.split(val_pct=0.2)
    predictor = RegressionPredictor(data, 'ridge', alpha=0.001)
    assert len(predictor.model) == len(data.links)
    assert predictor.model['4478'].get_params()['alpha'] == 0.001

    predictor = RegressionPredictor(data, 'svr', C=0.1, epsilon=0.01)
    assert len(predictor.model) == len(data.links)
    assert predictor.model['4478'].get_params()['C'] == 0.1
    assert predictor.model['4478'].get_params()['epsilon'] == 0.01

    predictor = RegressionPredictor(data, 'rf', n_estimators=100, max_depth=5)
    assert len(predictor.model) == len(data.links)
    assert predictor.model['4478'].get_params()['n_estimators'] == 100
    assert predictor.model['4478'].get_params()['max_depth'] == 5


def test_remove_invalid_data():
    net = SUMONetwork('tests/test_cases/network_1.xml')
    test_data = TrafficData(net, 10)
    test_data.features = ['speed']
    test_data.links = net.link_ids
    test_data.timestamps = list(range(100))
    test_data.data = np.array([[range(1000, 1100), range(2000, 2100)]])
    test_data.split(test_pct=0.1)
    predictor = RegressionPredictor(test_data, 'ridge', past_steps=1, alpha=0.001)
    data, label = predictor.get_neighborhood_data('train', 'A')
    data = np.mod(data, 10)
    label = np.broadcast_to(np.mod(label, 10), data.shape)
    assert_array_less(data, label)
