import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from src.prediction_models.prediction_model import PredictionModel


# TODO: add support for multiple output features
class RegressionPredictor(PredictionModel):
    """
    Prediction with sklearn regression models

    Parameters
    ----------
    traffic_data : TrafficData
        traffic data to load
    model_type : {'ridge', 'svr', 'rf'}
        type of regression model,
        'ridge' for ridge regression, 'svr' for support vector regression, 'rf' for random forest regression
    n_hops : int
        how many neighbors to include as input features
    past_steps : int
        how many previous time steps for target location to include as input features
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
    regressor_args
        hyperparameters to sklearn regression models,
        example: C and epsilon in SVR

    Attributes
    ----------
    model_type : str
        type of regression model
    n_hops : int
        how many neighbors to include as input features
    past_steps : int
        how many previous time steps for target location to include as input features
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
    regressor_args
        hyperparameters to sklearn regression models
    model
        the sklearn regression model
    """

    def __init__(self, traffic_data, model_type, n_hops=0, past_steps=0, horizon=1, source_feature=None, target_feature=None, direction='both',
                 turn_types='all', broken_prob=0.0, batch_save=False, batch_size=None, save_path=None, timestamp_feature=False, **regressor_args):
        super().__init__(traffic_data, source_feature=source_feature, target_feature=target_feature, horizon=horizon, broken_prob=broken_prob)

        self.model_type = model_type
        self.n_hops = n_hops
        self.past_steps = past_steps
        self.direction = direction
        self.turn_types = turn_types
        self.score = None
        self.timestamp_feature = timestamp_feature
        self.regressor_args = regressor_args

        if batch_save and not batch_size or batch_save and not save_path:
            raise ValueError
        elif batch_save:
            self.batch_save = batch_save
            self.batch_size = batch_size
            self.save_path = save_path + self.__str__() + '/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)

        # creates the regression models
        self.create_model()

    def __str__(self):
        """
        Returns a string representation of the model

        Returns
        -------
        str
            The string representation of this model
        """
        if self.regressor_args:
            arg_str = "-".join([str(arg) + str(v) for arg, v in self.regressor_args.items()])
            arg_str = "-" + arg_str
        else:
            arg_str = ""

        return f'{self.model_type}-n{self.n_hops}-p{self.past_steps}' + arg_str

    def train(self):
        # iterate through every link
        for i, link in enumerate(self.traffic_data.links):
            # get the feature and regression target for this link
            data, label = self.get_neighborhood_data('train', link)

            # fit the model
            self.model[link] = self.model[link].fit(data, label)
            self.score[link] = self.model[link].score(data, label)

            if self.batch_save and (i + 1) % self.batch_size == 0:
                path = self.save_path + f'rf_{i + 1 - self.batch_size}.pickle'
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                self.create_model()

        if self.batch_save and len(self.traffic_data.links) % self.batch_size != 0:
            path = self.save_path + f'rf_{int(len(self.traffic_data.links) / self.batch_size) * self.batch_size}.pickle'
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)

    def evaluate(self, mode, links=None):
        if not links or self.batch_save:
            links = self.traffic_data.links
        x_true = []
        x_pred = []

        # iterate through every link
        for i, link in enumerate(links):
            if self.batch_save and i % self.batch_size == 0:
                path = self.save_path + f'rf_{i}.pickle'
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)

            # get the feature and regression target for this link
            data, label = self.get_neighborhood_data(mode, link)

            # append to output
            x_true.append(label)
            pred = self.model[link].predict(data)
            x_pred.append(pred)

        # convert list to numpy array
        x_true = np.array(x_true)
        x_pred = np.array(x_pred)

        if self.traffic_data.normalizer:
            if len(self.target_feature) != 1:
                raise NotImplementedError("Multiple target features not supported")

            x_true = self.traffic_data.normalizer[self.target_feature[0]].inverse_transform(x_true)
            x_pred = self.traffic_data.normalizer[self.target_feature[0]].inverse_transform(x_pred)

        return x_true, x_pred

    def get_model_type(self):
        """
        Get the corresponding sklearn regression model from model_type

        Raises
        ------
        ValueError
            If the model type is invalid
        """
        if self.model_type == 'ridge':
            model = Ridge
        elif self.model_type == 'svr':
            model = SVR
        elif self.model_type == 'rf':
            model = RandomForestRegressor
            self.regressor_args['n_jobs'] = -2
        elif self.model_type == 'gbr':
            model = GradientBoostingRegressor
        else:
            raise ValueError('Error: model type not valid.')

        return model

    def create_model(self):
        """
        Creates the corresponding sklearn regression model into a dictionary
        """

        model = self.get_model_type()
        self.model = {link: model(**self.regressor_args) for link in self.traffic_data.links}
        self.score = {link: 0 for link in self.traffic_data.links}

    def get_neighborhood_data(self, dataset, target_link, as_df=False):
        """
        Get data from the neighbors of target_link

        Parameters
        ----------
        dataset : str
            which dataset to use, can be either ['train', 'val', 'test']
        target_link : str
            which target link to extract neighborhood data for
        as_df
            whether to convert output to pandas DataFrame

        Returns
        -------
        data : pandas.DataFrame
        label : pandas.DataFrame

        """
        dataset = self.get_dataset(dataset)
        neighbors = self.get_neighbors(target_link)

        # how many time steps need to be removed from the beginning of labels
        # example: using current plus 2 previous time steps to calculate next time step
        # time 0-2 -> 3, 1-3 -> 4 etc.
        # 2+1 data points are "lost" and need to be truncated
        horizon = self.horizon if isinstance(self.horizon, int) else max(self.horizon)
        truncate = self.past_steps + horizon

        # select the data from dataset
        neighborhood_data = dataset.loc[self.source_feature, neighbors, :].data
        # unroll data to 2D
        neighborhood_data = np.reshape(neighborhood_data, (-1, neighborhood_data.shape[-1]))
        target_data = dataset.loc[self.source_feature, target_link, :].data

        label_data = [dataset.sel(feature=self.target_feature, link=target_link).isel(time=slice(truncate, None)).data]
        # slide window back
        for i in range(1, horizon):
            label_data.insert(0, dataset.sel(feature=self.target_feature, link=target_link).isel(time=slice(truncate - i, -i)).data)

        label_data = np.concatenate(label_data, axis=0)
        label = label_data.T

        # 1 data point is always lost at the end since target_link needs to be offset by 1
        # data is a sliding window with size = number of data points
        # the for loop slides this window back in time and concatenate the time-shifted data to the dataset
        # example: get data starting from time = t3, then shift window back to t2 then concat data
        #       data                label
        # [[t3, t2, t1, t0],        [[t4],
        #  [t4, t3, t2, t1],         [t5],
        #  [t5, t4, t3, t2],         [t6],
        #       ...                   ...
        #  [t8, t7, t6, t5],         [t9],

        data = []
        cols = []

        # slide window back
        for i in range(self.past_steps + 1):
            data.append(target_data[:, i:-truncate + i])
            data.append(neighborhood_data[:, i:-truncate + i])
            if as_df:
                cols.extend([f'{target_link}_{feature}_past_{self.past_steps - i}' for feature in self.source_feature])
                cols.extend([f'{link}_{feature}_past{self.past_steps - i}' for feature in self.source_feature for link in neighbors])

        data = np.concatenate(data, axis=0)
        data = data.T

        if self.timestamp_feature:
            ts_features = self.get_timestamp_features(dataset.time.data[:-truncate])
            data = np.concatenate([data, ts_features.to_numpy()], axis=1)
            if as_df:
                cols.extend(ts_features.columns)

        # remove invalid data between multiple simulation runs
        if self.traffic_data.run_len:
            data, label = self.remove_invalid_data(data, label)
        if self.broken_prob:
            data = self.add_broken(data)

        # select only the desired regression targets for discrete time step horizons
        if isinstance(self.horizon, list):
            if as_df:
                label_cols = [f'{feature}_horizon_{i}' for feature in self.target_feature for i in self.horizon]
            horizon = [i - 1 for i in self.horizon]
            label = np.take(label, horizon, axis=1)

        elif as_df:
            label_cols = [f'{feature}_horizon_{i}' for feature in self.target_feature for i in range(1, horizon + 1)]

        label = np.squeeze(label)

        if as_df:
            label = pd.DataFrame(label, columns=label_cols)
            data = pd.DataFrame(data, columns=cols)

        return data, label

    def remove_invalid_data(self, data, label):
        """
        Removes data points to get rid of data, label combo that span 2 different Aimsun simulation runs

        Parameters
        ----------
        data : numpy.array
            The data to be processed
        label : numpy.array
            The label to be processed

        Returns
        -------
        data : numpy.array
            The processed data
        label : numpy.array
            The process label
        """
        horizon = self.horizon if isinstance(self.horizon, int) else max(self.horizon)  # the last predicted time step
        truncate = self.past_steps + horizon
        remove = []
        for i in range(truncate):
            remove.extend(list(range(self.traffic_data.run_len - i - 1, len(data), self.traffic_data.run_len)))
        remove = sorted(remove)

        data = np.delete(data, remove, axis=0)
        label = np.delete(label, remove, axis=0)

        return data, label

    def get_neighbors(self, target_link):
        """
        Get all the links within n hops from target by breadth first search with n iterations

        Parameters
        ----------
        target_link : str
            The target link to start the search from

        Returns
        -------
        list of str
            The links within the neighborhood of target, excluding target itself
        """
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
        return neighbors

    def get_timestamp_features(self, timestamps):
        known_holidays = ['2018-01-01', '2018-01-15', '2018-02-19', '2016-07-04']
        known_holidays = [pd.Timestamp(i) for i in known_holidays]

        time_df = pd.DataFrame(timestamps, columns=['time'])

        holiday_df = pd.get_dummies((time_df['time'].dt.weekday > 4) | (time_df['time'].dt.date.apply(lambda date: date in known_holidays)))
        holiday_df.columns = ['workday', 'holiday']

        day_hour_df = pd.get_dummies(time_df['time'].dt.dayofweek * 24 + time_df['time'].dt.hour)
        day_hour_df.columns = [f'weekday_{d}_hour_{h}' for d in range(7) for h in range(24)]

        out_df = pd.concat([day_hour_df, holiday_df], axis=1)
        return out_df
