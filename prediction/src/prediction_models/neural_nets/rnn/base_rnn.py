from src.prediction_models.neural_nets.nn_predictor import NNPredictor


class BaseRNNPredictor(NNPredictor):
    """
    Predictor using RNNs

    Parameters
    ----------
    traffic_data : TrafficData
        Traffic data to load
    hidden_size : int
        The size of hidden layer
    batch_size : int
        The size of each training batch
    seq_len : int
        The length of each sequence
    model_type : {'gru', 'lstm'}
        The type of model
    max_epoch : int
        when to stop training
    learning_rate : float
    target_feature : list of str
        what features to predict, default: ['speed']
    horizon : int or list of int
        which horizons to predict, default 1
        if int is provided, prediction is generated for every horizon up to the number provided
        if list is provided, only prediction for the specified horizons are generated
    predictor_method : {'rolling', 'separate'}
        if horizon is a list, specify the method for the prediction side of RNN
        'rolling' means every horizon is predicted internally but only the specified horizons are used
        'separate' means that only the specified horizons are predicted, and they are predicted independently
    save : bool
        whether to save model
    save_path : str
        where to save the model
    **optimizer_args
        Other keyword arguments to the PyTorch optimizer
    """

    def __init__(self, traffic_data, hidden_size, batch_size, seq_len, model_type, max_epoch=5, learning_rate=1e-4,
                 target_feature=None, horizon=1, save=False, save_path='../models/gru/', save_interval=50, **optimizer_args):
        super().__init__(traffic_data, batch_size=batch_size, seq_len=seq_len, max_epoch=max_epoch, learning_rate=learning_rate, horizon=horizon,
                         target_feature=target_feature, save=save, save_path=save_path, save_interval=save_interval, **optimizer_args)

        self.hidden_size = hidden_size
        self.model_type = model_type

    def __str__(self):
        raise NotImplementedError

    def train(self):
        super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)
