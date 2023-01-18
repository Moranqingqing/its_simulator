import os
import pickle

import numpy as np
from pmdarima.arima import auto_arima

from src.prediction_models.prediction_model import PredictionModel


# TODO: add support for multiple features
class ARIMAPredictor(PredictionModel):
    """
    Predictor with autoregressive integrated moving average (ARIMA) model
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima_model.ARIMA

    Parameters
    ----------
    traffic_data : TrafficData
        traffic data to load
    target_feature : list of str
        what features to predict,
        default: ['speed']
    order : 3-tuple of int
        the order of ARIMA model
    """

    def __init__(self, traffic_data, max_p=1, target_feature=None, horizon=1, broken_prob=0.0, batch_save=False, batch_size=None, save_path=None):
        super().__init__(traffic_data, target_feature=target_feature, horizon=horizon, broken_prob=broken_prob)

        self.max_p = max_p
        self.model = {link: None for link in self.traffic_data.links}

        if batch_save and not batch_size or batch_save and not save_path:
            raise ValueError
        elif batch_save:
            self.batch_save = batch_save
            self.batch_size = batch_size
            self.save_path = save_path + self.__str__() + '/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

    def __str__(self):
        return f"ARIMA({self.max_p})"

    def train(self):
        if len(self.target_feature) > 1:
            raise NotImplementedError("More than 1 target features not supported")
        for i, link in enumerate(self.traffic_data.links):
            data = self.traffic_data.train_data.sel(feature=self.target_feature, link=link).data
            data = data.squeeze()
            self.model[link] = auto_arima(data, max_p=self.max_p, trace=False, error_action='ignore', suppress_warnings=True)

            if self.batch_save and (i + 1) % self.batch_size == 0:
                path = self.save_path + f'arima_{i + 1 - self.batch_size}.pickle'
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                self.model = {link: None for link in self.traffic_data.links}

        if self.batch_save and len(self.traffic_data.links) % self.batch_size != 0:
            path = self.save_path + f'arima_{int(len(self.traffic_data.links) / self.batch_size) * self.batch_size}.pickle'
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)

    def evaluate(self, mode, links=None):

        # predict all links if not specified
        if not links or self.batch_save:
            links = self.traffic_data.links

        dataset = self.get_dataset(mode)

        x_true = []
        x_pred = []

        # iterate through all links
        for i, link in enumerate(links):
            if self.batch_save and i % self.batch_size == 0:
                path = self.save_path + f'arima_{i}.pickle'
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)

            data = dataset.sel(feature=self.target_feature, link=link).data
            data = data.squeeze()

            link_true = []
            link_pred = []

            # the last predicted time step
            horizon = self.horizon if isinstance(self.horizon, int) else max(self.horizon)

            # rolling prediction
            last_broken = 0
            for j in range(len(data) - horizon):
                link_true.append(data[j + 1:j + horizon + 1])
                new_res = self.model[link].arima_res_.apply(data[:j + 1 - last_broken])
                link_pred.append(new_res.forecast(horizon + last_broken)[last_broken:])
                if np.random.random() < self.broken_prob:
                    last_broken += 1
                else:
                    last_broken = 0

            link_true = np.array(link_true)
            link_pred = np.array(link_pred)

            x_true.append(link_true)
            x_pred.append(link_pred)

        x_true = np.array(x_true)
        x_pred = np.array(x_pred)

        # select the discrete horizons
        if isinstance(self.horizon, list):
            horizon = [i - 1 for i in self.horizon]
            x_true = np.take(x_true, horizon, axis=2)
            x_pred = np.take(x_pred, horizon, axis=2)

        return x_true, x_pred

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
