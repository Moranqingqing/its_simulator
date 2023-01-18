import pickle

import numpy as np

from src.prediction_models.regression import RegressionPredictor


# TODO: add support for multiple output features
class MultiRegressionPredictor(RegressionPredictor):
    """
    Prediction with multiple sklearn regression models, divided based on change in target feature over past time steps and a threshold

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
    threshold : int
        The threshold change in target feature to divide the sample
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

    def __init__(self, traffic_data, model_type, n_hops=0, past_steps=0, threshold=5, horizon=1, source_feature=None, target_feature=None,
                 direction='both', turn_types='all', broken_prob=0.0, batch_save=False, batch_size=None, save_path=None, **regressor_args):

        self.threshold = threshold
        super().__init__(traffic_data, model_type=model_type, n_hops=n_hops, past_steps=past_steps, horizon=horizon,
                         source_feature=source_feature, target_feature=target_feature, direction=direction, turn_types=turn_types,
                         broken_prob=broken_prob, batch_save=batch_save, batch_size=batch_size, save_path=save_path, **regressor_args)

        self.samples = None

        if len(self.target_feature) > 1:
            raise NotImplementedError("Multiple target feature not supported")
        if self.target_feature[0] not in self.source_feature:
            raise NotImplementedError("Target feature must be in source features")

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

        # return f'{self.model_type}multi-n{self.n_hops}-p{self.past_steps}-t{self.threshold}' + arg_str
        return f'{self.model_type}multi-n{self.n_hops}-p{self.past_steps}' + arg_str

    def train(self):
        # iterate through every link
        for i, link in enumerate(self.traffic_data.links):
            # get the feature and regression target for this link
            data, label = self.get_neighborhood_data('train', link, as_df=True)
            low, mid, high = self.get_threshold_chunks(data, link)

            # fit the different models for target feature decreasing/increasing over threshold
            for c, chunk in enumerate([low, mid, high]):
                if data[chunk].shape[0] > 0:
                    if label[chunk].shape[1] == 1:
                        self.model[link][c] = self.model[link][c].fit(data[chunk].to_numpy(), np.squeeze(label[chunk].to_numpy(), axis=1))
                        self.score[link][c] = self.model[link][c].score(data[chunk].to_numpy(), np.squeeze(label[chunk].to_numpy(), axis=1))
                    else:
                        self.model[link][c] = self.model[link][c].fit(data[chunk].to_numpy(), label[chunk].to_numpy())
                        self.score[link][c] = self.model[link][c].score(data[chunk].to_numpy(), label[chunk].to_numpy())
                self.samples[link][c] = data[chunk].shape[0]

            if self.batch_save and (i + 1) % self.batch_size == 0:
                path = self.save_path + f'rf_multi_{i + 1 - self.batch_size}.pickle'
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                self.create_model()

        if self.batch_save and len(self.traffic_data.links) % self.batch_size != 0:
            path = self.save_path + f'rf_multi_{int(len(self.traffic_data.links) / self.batch_size) * self.batch_size}.pickle'
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
                path = self.save_path + f'rf_multi_{i}.pickle'
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)

            # get the feature and regression target for this link
            data, label = self.get_neighborhood_data(mode, link, as_df=True)
            low, mid, high = self.get_threshold_chunks(data, link)

            # append to output
            x_true.append(np.concatenate([label[low], label[mid], label[high]], axis=0))

            pred = []
            for c, chunk in enumerate([low, mid, high]):
                if data[chunk].shape[0] > 0:
                    if self.samples[link][c] == 0:
                        pred.append(self.model[link][1].predict(data[chunk]))
                    else:
                        pred.append(self.model[link][c].predict(data[chunk]))
            x_pred.append(np.concatenate(pred, axis=0))

        # convert list to numpy array
        x_true = np.squeeze(np.array(x_true))
        x_pred = np.squeeze(np.array(x_pred))

        if self.traffic_data.normalizer:
            if len(self.target_feature) != 1:
                raise NotImplementedError("Normalizing multiple target features not supported")

            x_true = self.traffic_data.normalizer[self.target_feature[0]].inverse_transform(x_true)
            x_pred = self.traffic_data.normalizer[self.target_feature[0]].inverse_transform(x_pred)

        return x_true, x_pred

    def get_threshold_chunks(self, data, link):
        """
        Given a DataFrame of samples, separate the samples into three categories based on the amount of change over the target feature in past_steps

        Parameters
        ----------
        data : pandas.DataFrame
            The DataFrame object to divide
        link : str
            The current target link name, used to separate the features that belong to the target link from neighboring links

        Returns
        -------
        low : pandas.DataFrame
            True where change in feature is below negative threshold
        mid : pandas.DataFrame
            True where change in feature is between negative and positive threshold
        high : pandas.DataFrame
            True where change in feature is above threshold
        """
        low = data[f'{link}_{self.target_feature[0]}_past_0'] - data[f'{link}_{self.target_feature[0]}_past_{self.past_steps}'] < -self.threshold
        mid = np.abs(data[f'{link}_{self.target_feature[0]}_past_0'] -
                     data[f'{link}_{self.target_feature[0]}_past_{self.past_steps}']) <= self.threshold
        high = data[f'{link}_{self.target_feature[0]}_past_0'] - data[f'{link}_{self.target_feature[0]}_past_{self.past_steps}'] > self.threshold

        return low, mid, high

    def create_model(self):
        """
        Creates the corresponding sklearn regression model into a dictionary
        """

        model = self.get_model_type()
        self.model = {link: [model(**self.regressor_args) for _ in range(3)] for link in self.traffic_data.links}
        self.score = {link: [0 for _ in range(3)] for link in self.traffic_data.links}
        self.samples = {link: [0 for _ in range(3)] for link in self.traffic_data.links}
