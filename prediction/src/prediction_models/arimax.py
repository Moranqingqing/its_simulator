import datetime
import pickle

import numpy as np
import pandas
from statsmodels.tsa.arima.model import ARIMA

from src.prediction_models.prediction_model import PredictionModel


# TODO: add support for multiple features
class ARIMAXPredictor(PredictionModel):
    """
    ARIMA with explanatory variables

    Parameters
    ----------
    traffic_data : TrafficData
        traffic data to load
    n_hops : int
        how many neighbors to include as input features
    direction : str
        which direction of target are neighbors selected,
        choices: ['upstream', 'downstream', 'both'],
        default: 'both'
    turn_types : str
        which turn types to include as neighbors,
        choices: ['straight', 'all'],
        default: 'all'
    target_feature : list of str
        what features to predict,
        default: ['speed']
    order : 3-tuple of int
        the order of ARIMA model

    Attributes
    ----------
    n_hops : int
        how many neighbors to include as input features
    target_feature : list of str
        what features to predict,
        default: ['speed']
    direction : str
        which direction of target are neighbors selected,
        choices: ['upstream', 'downstream', 'both'],
        default: 'both'
    turn_types : str
        which turn types to include as neighbors,
        choices: ['straight', 'all'],
        default: 'all'
    model
        the statsmodel SARIMAX model
    results
        the statsmodel SARIMAX results after training
    """

    def __init__(self, traffic_data, n_hops, direction='both', turn_types='all', target_feature=None,
                 horizon=1, order=(1, 0, 1)):
        super().__init__(traffic_data, target_feature=target_feature, horizon=horizon)
        self.order = order
        self.n_hops = n_hops
        self.direction = direction
        self.turn_types = turn_types
        self.model = {link: None for link in self.traffic_data.links}
        self.results = {link: None for link in self.traffic_data.links}

    def __str__(self):
        return "arimax-n{}-{}-{}-{}".format(self.n_hops, self.order[0], self.order[1], self.order[2])

    def train(self):
        for link in self.traffic_data.links:
            endog, exog = self.get_neighborhood_data('train', link)
            # statsmodels allows timestamps to be used in ARIMA models, probably not needed
            if type(self.traffic_data.timestamps[0]) is datetime.datetime or type(
                    self.traffic_data.timestamps[0]) is pandas.Timestamp:
                self.model[link] = ARIMA(endog, exog, order=self.order, dates=self.traffic_data.train_timestamps)
            else:
                self.model[link] = ARIMA(endog, exog, order=self.order)

            self.results[link] = self.model[link].fit()
            self.results[link].remove_data()

    def evaluate(self, mode, links=None):
        dataset = self.get_dataset(mode)
        if not links:
            links = self.traffic_data.links
        feature_indices = [self.traffic_data.features.index(feature) for feature in self.target_feature]
        x_true = []
        x_pred = []
        for link in links:
            link_indices = self.traffic_data.links.index(link)
            data = dataset[feature_indices, link_indices, :]
            data = data.squeeze()
            endog, exog = self.get_neighborhood_data(mode, link)

            link_true = []
            link_pred = []

            horizon = self.horizon if isinstance(self.horizon, int) else max(self.horizon)
            for i in range(len(data) - horizon):
                link_true.append(data[i + 1:i + horizon + 1])
                new_res = self.results[link].apply(endog[:i + 1], exog=exog[:i + 1])
                link_pred.append(new_res.forecast(horizon))
            link_true = np.array(link_true)
            link_pred = np.array(link_pred)

            x_true.append(link_true)
            x_pred.append(link_pred)

        x_true = np.array(x_true)
        x_pred = np.array(x_pred)

        if isinstance(self.horizon, list):
            horizon = [i - 1 for i in self.horizon]
            x_true = np.take(x_true, horizon, axis=2)
            x_pred = np.take(x_pred, horizon, axis=2)

        return x_true, x_pred

    def get_neighborhood_data(self, mode, target_link):
        dataset = self.get_dataset(mode)
        upstream = True if self.direction == 'upstream' or self.direction == 'both' else False
        downstream = True if self.direction == 'downstream' or self.direction == 'both' else False
        neighbors = [target_link]
        # breadth first search with n iterations
        for _ in range(self.n_hops):
            new_neighbors = [n for n in neighbors]
            for neighbor in neighbors:
                # add upstream connections
                if upstream:
                    if self.turn_types == 'all':
                        new_neighbors.extend([link for link in self.traffic_data.network.upstream[neighbor]['left']])
                        new_neighbors.extend([link for link in self.traffic_data.network.upstream[neighbor]['right']])
                    new_neighbors.extend([link for link in self.traffic_data.network.upstream[neighbor]['straight']])
                # add downstream connections
                if downstream:
                    if self.turn_types == 'all':
                        new_neighbors.extend([link for link in self.traffic_data.network.downstream[neighbor]['left']])
                        new_neighbors.extend([link for link in self.traffic_data.network.downstream[neighbor]['right']])
                    new_neighbors.extend([link for link in self.traffic_data.network.downstream[neighbor]['straight']])
            new_neighbors = list(set(new_neighbors))
            neighbors = new_neighbors
        # remove target_link itself from neighbors
        neighbors.remove(target_link)

        # get the link indices for all neighbors
        neighbors_index = np.where(np.isin(self.traffic_data.links, neighbors))[0]
        # link index for the target_link
        target_index = self.traffic_data.links.index(target_link)
        # feature indices for the features interested
        feature_index = np.where(np.isin(self.traffic_data.features, self.target_feature))[0]

        # select the traffic_data from np array
        neighborhood_data = dataset[:, neighbors_index, :]
        # unroll traffic_data to 2D
        neighborhood_data = np.reshape(neighborhood_data, (-1, neighborhood_data.shape[-1]))
        target_non_feature_data = np.delete(dataset[:, target_index, :], feature_index, axis=0)

        exog = np.concatenate((neighborhood_data, target_non_feature_data), axis=0)
        exog = exog.T

        endog = dataset[feature_index, target_index, :]
        endog = endog.T

        return endog, exog

    def save(self, path):
        self.remove_data()
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def remove_data(self):
        """
        Removes the data in the ARIMAX models. Called before saving to reduce size on disk.
        """
        for link in self.traffic_data.links:
            self.results[link].remove_data()
