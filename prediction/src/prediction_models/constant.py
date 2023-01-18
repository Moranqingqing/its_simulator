import numpy as np

from src.prediction_models.prediction_model import PredictionModel


class ConstantPredictor(PredictionModel):
    """
    Predictor that always predicts the same constant value for a given link/feature,
    by predicting the average across the time axis
    """

    def __init__(self, traffic_data, target_feature=None, horizon=1):
        super().__init__(traffic_data, target_feature=target_feature, horizon=horizon)

    def __str__(self):
        return "constant"

    def train(self):
        """
        Create the model by taking average across the time axis

        Returns
        -------
        None
        """
        # 2 * N
        self.model = np.average(self.traffic_data.train_data[:, :, :], axis=2)

    def evaluate(self, mode, links=None):
        dataset = self.get_dataset(mode)
        timestamps = self.get_timestamps(mode)
        if not links:
            links = self.traffic_data.links

        # get indices for flow/speed, and link.
        # traffic_data.features = ['flow', 'speed'], target_feature = ['speed']
        feature_indices = np.where(np.isin(self.traffic_data.features, self.target_feature))[0]
        link_indices = np.where(np.isin(self.traffic_data.links, links))[0]

        # slice along target feature and links
        x_true = dataset.sel(feature=self.target_feature, link=links)

        # np.take to preserve axis
        model = self.model.sel(feature=self.target_feature, link=links)

        # make the two arrays the same shape
        x_pred = np.repeat(model[:, :, np.newaxis], dataset.data.shape[-1], axis=2)

        return x_true, x_pred
