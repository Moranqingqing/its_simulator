import pickle

import numpy as np
from numpy.testing import *

from src.network.network_model import NetworkModel
from src.network.sumo_network import SUMONetwork
from src.traffic_data.aimsun_traffic_data import AimsunTrafficData
from src.traffic_data.csv_traffic_data import CSVTrafficData


def test_csv_1():
    net = SUMONetwork('tests/test_cases/network_3.xml')
    data = CSVTrafficData('tests/test_cases/csv_data_1.csv', net)
    assert data.features == ['hi']
    assert data.links == ['A']
    assert data.timestamps == ['1']
    assert_array_equal(data.data, [[[5]]])
    data.split()
    assert_array_equal(data.train_data, data.data)
    assert data.val_data.size == 0
    assert data.test_data.size == 0


def test_csv_2():
    net = SUMONetwork('tests/test_cases/network_1.xml')
    data = CSVTrafficData('tests/test_cases/csv_data_2.csv', net)
    assert data.features == ['speed', 'flow']
    assert data.links == ['A', 'B']
    assert data.timestamps == ['1', '2']
    assert_array_equal(data.data, [[[50.2, 45.8], [30.8, 52.6]],
                                   [[20, 40], [52, 78]]])
    data.split()
    assert_array_equal(data.train_data, data.data)
    assert data.val_data.size == 0
    assert data.test_data.size == 0


def test_split():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    data.split(val_pct=0.2, test_pct=0.2)

    # int(0.2*48) is 9, there should be 9 samples in both validation and test set
    assert data.train_timestamps == [str(i) for i in range(1, 31)]
    assert data.val_timestamps == [str(i) for i in range(31, 40)]
    assert data.test_timestamps == [str(i) for i in range(40, 49)]

    assert_array_equal(data.train_data, data.data[:, :, :30])
    assert_array_equal(data.val_data, data.data[:, :, 30: 39])
    assert_array_equal(data.test_data, data.data[:, :, 39:])


def test_aimsun():
    net = NetworkModel().load('tests/test_cases/qew/qew_net.json')
    data_csv = CSVTrafficData('tests/test_cases/qew/qew.csv', net)
    data_sqlite = AimsunTrafficData('tests/test_cases/qew/basecase5min.sqlite', net)
    assert data_csv.links == data_sqlite.links
    assert data_csv.timestamps == data_sqlite.timestamps
    assert data_sqlite.features == ['flow', 'speed']
    feature_indices = np.where(np.isin(data_csv.features, ['flow', 'speed']))[0]
    assert_array_equal(data_csv.data[feature_indices, :, :], data_sqlite.data)


def test_split_random():
    with open('tests/test_cases/qew/50runs-1min.pickle', 'rb') as f:
        data = pickle.load(f)
    data.split_random(test_pct=0.2)
    assert len(data.test_timestamps) == 2400
    assert not data.val_timestamps
    assert len(data.train_timestamps) == 9600
    assert sorted(data.test_timestamps + data.train_timestamps) == sorted(data.timestamps)

    assert not data.val_data
    assert data.test_data.shape == (2, 56, 2400)
    assert data.train_data.shape == (2, 56, 9600)
    concat = np.concatenate([data.test_data, data.train_data], axis=2)
    assert concat.shape == data.data.shape
    assert_array_equal(np.sort(concat), np.sort(data.data))
