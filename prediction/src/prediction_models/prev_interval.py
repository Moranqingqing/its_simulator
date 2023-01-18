import numpy as np

from src.prediction_models.prediction_model import PredictionModel


class PreviousIntervalPredictor(PredictionModel):
    """
    Predictor that predicts the same value as the last time step
    """

    def __init__(self, traffic_data, target_feature=None, horizon=1, broken_prob=0.0):
        super().__init__(traffic_data, target_feature=target_feature, horizon=horizon, broken_prob=broken_prob)

    def __str__(self):
        """
        Returns a string representation of the model

        Returns
        -------
        str
            The string representation of this model
        """
        return "previous interval"

    def train(self):
        # no training required for this model
        pass

    def evaluate(self, mode, links=None):
        dataset = self.get_dataset(mode).data
        if not links:
            links = self.traffic_data.links
        feature_indices = np.where(np.isin(self.traffic_data.features, self.target_feature))[0]
        link_indices = np.where(np.isin(self.traffic_data.links, links))[0]

        horizon = self.horizon if isinstance(self.horizon, int) else max(self.horizon)
        x_true = [dataset[feature_indices[:, None], link_indices[None, :], horizon:]]
        for i in range(1, horizon):
            x_true.insert(0, dataset[feature_indices[:, None], link_indices[None, :], horizon - i:-i])
        x_true = np.stack(x_true, axis=-1)

        x_pred = dataset[feature_indices[:, None], link_indices[None, :], :-horizon]
        x_pred = np.repeat(x_pred[..., np.newaxis], horizon, axis=-1)

        if self.traffic_data.run_len:
            self.remove_invalid_data(x_true, x_pred)
        if self.broken_prob:
            for i in range(1, x_pred.shape[-2]):
                if np.random.random() < self.broken_prob:
                    idx = [slice(None)] * x_pred.ndim
                    idx[-2] = i
                    x_pred[tuple(idx)] = np.take(x_pred, i - 1, -2)

        # select the discrete horizons
        if isinstance(self.horizon, list):
            horizon = [i - 1 for i in self.horizon]
            x_true = np.take(x_true, horizon, -1)
            x_pred = np.take(x_pred, horizon, -1)

        return x_true, x_pred

    def remove_invalid_data(self, x_true, x_pred):
        """
        Removes data points to get rid of data, label combo that span 2 different runs

        Parameters
        ----------
        x_true : numpy.array
            The traffic_data to be processed
        x_pred : numpy.array
            The label to be processed

        Returns
        -------
        x_true : numpy.array
            The processed traffic_data
        x_pred : numpy.array
            The process label
        """
        remove = [range(self.traffic_data.run_len - 1, len(x_true), self.traffic_data.run_len)]
        remove = sorted(remove)

        x_true = np.delete(x_true, remove, axis=0)
        x_pred = np.delete(x_pred, remove, axis=0)

        return x_true, x_pred
