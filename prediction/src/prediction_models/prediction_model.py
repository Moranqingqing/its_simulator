import pickle

import numpy as np
import xarray as xr


class PredictionModel:
    """
    Base class for all SOW4 prediction models

    Parameters
    ----------
    traffic_data : TrafficData
        Traffic data to load
    source_feature : list of str
        what features to input, default: ['flow', 'speed']
    target_feature : list of str
        What features to predict (default is ['speed'], which predicts speed only)
    horizon : int or list of int
        The prediction horizon of this prediction model, in terms of number of time steps. Integer represents prediction horizon including every time
        step up to the integer. List represents prediction horizon of discrete time steps. (default is 1, predicts next time step only)
    broken_prob : float
        The probability that the data is missing (default is 0)

    Attributes
    ----------
    traffic_data : TrafficData
        The loaded traffic data
    source_feature : list of str
        what features to input, default: ['flow', 'speed']
    target_feature : list of str
        What features to predict, (default is ['speed'], which predicts speed only)
    horizon : int or list of int
        The prediction horizon of this prediction model, in terms of number of time steps. Integer represents prediction horizon including every time
        step up to the integer. List represents prediction horizon of discrete time steps. (default is 1, predicts next time step only)
    broken_prob : float
        The probability that the data is missing
    model
        The model that the SOW4 model uses
    """

    def __init__(self, traffic_data, source_feature=None, target_feature=None, horizon=1, broken_prob=0.0):
        self.traffic_data = traffic_data
        if source_feature is None:
            self.source_feature = ['flow', 'speed']
        else:
            self.source_feature = sorted(source_feature, key=lambda e: self.traffic_data.features.index(e))

        if target_feature is None:
            self.target_feature = ['speed']
        else:
            # sort the target features list so that it is in order for indexing later
            self.target_feature = sorted(target_feature, key=lambda e: self.traffic_data.features.index(e))
        self.horizon = horizon
        self.broken_prob = broken_prob
        self.model = None

        self.prediction_data = None

    def train(self):
        """
        Trains the model on training set traffic_data

        Returns
        -------
        None
        """
        raise NotImplementedError

    def evaluate(self, mode, links=None):
        """
        Evaluates the model using the test set

        Parameters
        ----------
        mode : {'val', 'test'}
            which dataset to evaluate on
        links : list of str
            Which links to include in the evaluation,
            defaults to all links

        Returns
        -------
        x_true : numpy.array
            The true value of the test set
        x_pred : numpy.array
            The predicted values
        """
        raise NotImplementedError

    def predict(self, links=None):
        """
        Generate predictions

        Parameters
        ----------
        links : list of str
            Which links to include in the prediction,
            defaults to all links

        Returns
        -------
        pred : numpy.array
            The predicted values
        """
        raise NotImplementedError

    def save(self, path):
        """
        Saves the model

        Parameters
        ----------
        path : str
            The folder path that this model will be saved in

        Returns
        -------
        None
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def get_dataset(self, dataset):
        """
        Get the corresponding dataset from traffic_data

        Parameters
        ----------
        dataset : {'train', 'val', 'test'}
            The string corresponding to the desired dataset

        Raises
        ------
        ValueError

        Returns
        -------
        np.array
            The desired dataset
        """
        if dataset == 'train':
            return self.traffic_data.train_data
        elif dataset == 'val':
            return self.traffic_data.val_data
        elif dataset == 'test':
            return self.traffic_data.test_data
        else:
            raise ValueError

    def get_timestamps(self, dataset):
        """
        Get the corresponding dataset from traffic_data

        Parameters
        ----------
        dataset : {'train', 'val', 'test'}
            The string corresponding to the desired dataset

        Raises
        ------
        ValueError

        Returns
        -------
        list
            The desired timestamps
        """
        if dataset == 'train':
            return self.traffic_data.train_timestamps
        elif dataset == 'val':
            return self.traffic_data.val_timestamps
        elif dataset == 'test':
            return self.traffic_data.test_timestamps
        else:
            raise ValueError

    def add_broken(self, data):
        """
        Mask part of data in the dataset based on the probability of missing data

        Parameters
        ----------
        data : np.array
            The data array to be masked

        Returns
        -------
        np.array
            The masked data array

        """
        return np.where(np.random.random_sample(data.shape) < self.broken_prob, -1, data)

    def set_prediction_data(self, data):
        """
        Provide the model with the most up-to-date data to generate predictions from

        Parameters
        ----------
        data : TrafficData

        Returns
        -------
        None
        """
        self.prediction_data = data

    def append_prediction_data(self, data):
        """
        Update the model data with the most up-to-date data to generate predictions from

        Parameters
        ----------
        data : TrafficData

        Returns
        -------
        None
        """
        self.prediction_data.data = xr.concat([self.prediction_data.data, data], dim='time')
