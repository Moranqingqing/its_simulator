import numpy as np
import pandas as pd
import xarray as xr

from src.prediction_models.prediction_model import PredictionModel
from src.traffic_data.aimsun_traffic_data import AimsunTrafficData


class HistoricalAveragePredictor(PredictionModel):
    """
    Predict the value using historical average with defined period

    Parameters
    ----------
    traffic_data : TrafficData
        traffic data to load
    period : {'day', 'week'}
        The length of each period
    """

    def __init__(self, traffic_data, period='day', target_feature=None, horizon=1):
        super().__init__(traffic_data, target_feature=target_feature, horizon=horizon)

        self.period = period
        self.model_timestamps = []
        self.create_model_timestamps()
        self.model = np.empty(
            (len(self.traffic_data.features), len(self.traffic_data.links), len(self.model_timestamps)))

    def __str__(self):
        """
        Returns a string representation of the model

        Returns
        -------
        str
            The string representation of this model
        """
        if self.period == 'day':
            return "historical daily average"
        elif self.period == 'week':
            return "historical weekly average"

    def train(self):
        if isinstance(self.traffic_data, AimsunTrafficData):
            num_runs = int(len(self.traffic_data.timestamps) / self.traffic_data.run_len)
            # add extra dimension for each run
            data = self.traffic_data.data.data
            model = np.reshape(data, (data.shape[0], data.shape[1], num_runs, self.traffic_data.run_len))
            # take average across runs
            self.model = np.mean(model, axis=2)
            return

        # Historical average requires timestamps
        if type(self.traffic_data.timestamps[0]) is not pd.Timestamp:
            raise TypeError

        if self.period == 'day':
            self.model = xr.concat([v.groupby('time.time').mean() for i, v in self.traffic_data.train_data.groupby_bins("time.weekday", [0, 4, 7])],
                                   dim='weekday')
        elif self.period == 'week':
            self.model = xr.concat([v.groupby('time.time').mean() for i, v in self.traffic_data.train_data.groupby("time.weekday")], dim='weekday')
        else:
            return ValueError

    def evaluate(self, mode, links=None):
        dataset = self.get_dataset(mode).data
        if not links:
            links = self.traffic_data.links

        # get indices
        feature_indices = np.where(np.isin(self.traffic_data.features, self.target_feature))[0]
        link_indices = np.where(np.isin(self.traffic_data.links, links))[0]

        x_true = dataset
        x_true = x_true[feature_indices, link_indices, :]

        if isinstance(self.traffic_data, AimsunTrafficData):
            num_runs = int(x_true.shape[2] / self.traffic_data.run_len)
            x_pred = np.tile(self.model, (1, 1, num_runs))
            x_pred = x_pred[feature_indices, link_indices, :]

        else:  # HERE data
            if self.period == 'day':
                x_pred = np.stack([self.model.sel(feature=self.target_feature, link=links,
                                                  time=pd.Timestamp(t).time(), weekday=0 if pd.Timestamp(t).weekday() < 5 else 1).data
                                   for t in self.traffic_data.test_data.time.data], axis=2)
            else:
                x_pred = np.stack([self.model.sel(feature=self.target_feature, link=links,
                                                  time=pd.Timestamp(t).time(), weekday=pd.Timestamp(t).weekday()).data
                                   for t in self.traffic_data.test_data.time.data], axis=2)
            x_pred = np.squeeze(x_pred)

        return x_true, x_pred

    def create_model_timestamps(self):
        """
        Creates the lookup table for the historical average model
        """
        if isinstance(self.traffic_data, AimsunTrafficData):
            self.model_timestamps = list(range(self.traffic_data.run_len))
            return
        if self.period == 'day':
            # two columns, one for weekdays and one for weekend
            self.model_timestamps = list(dict.fromkeys(
                [(self.is_weekday(t.day_name()), t.time()) for t in self.traffic_data.timestamps]))
        elif self.period == 'week':
            # one column for each day of week
            self.model_timestamps = list(dict.fromkeys(
                [(t.day_name(), t.time()) for t in self.traffic_data.timestamps]))
        else:
            raise ValueError

    @staticmethod
    def is_weekday(day):
        """
        Determines whether the input day is a weekday

        Parameters
        ----------
        day : str
            The input day of week

        Returns
        -------
        str
            The boolean value representing whether the day is a weekday
        """
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            return 1
        else:
            return 0
