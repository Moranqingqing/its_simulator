import torch
import torch.nn as nn

from src.prediction_models.neural_nets.rnn.base_rnn import BaseRNNPredictor


class GlobalRNNModel(nn.Module):
    """
    The RNN model used in RNNPredictor

    Parameters
    ----------
    predictor : PredictionModel
        The predictor that contains this model

    Attributes
    ----------
    rnn
        The RNN model of the predictor
    fc
        The fully-connected network to generate prediction from hidden state
    """

    def __init__(self, predictor):
        super(GlobalRNNModel, self).__init__()

        self.predictor = predictor

        if self.predictor.model_type == 'lstm':
            self.rnn = nn.LSTMCell(self.predictor.input_size, self.predictor.hidden_size)
        elif self.predictor.model_type == 'gru':
            self.rnn = nn.GRUCell(self.predictor.input_size, self.predictor.hidden_size)
        else:
            raise ValueError
        self.fc = nn.Linear(self.predictor.hidden_size, self.predictor.output_size)

    def forward(self, x):
        # Set an initial hidden state
        hidden = torch.zeros(x.shape[0], self.predictor.hidden_size)
        hidden = hidden.to(self.predictor.device)

        if self.predictor.model_type == 'lstm':
            cell = torch.zeros(x.shape[0], self.predictor.hidden_size)
            cell = cell.to(self.predictor.device)

        # Forward propagate the RNN
        for i in range(self.predictor.seq_len):
            if self.predictor.model_type == 'lstm':
                hidden, cell = self.rnn(x[:, i, :], (hidden, cell))
            else:
                hidden = self.rnn(x[:, i, :], hidden)

        # Pass the RNN output to the fully connected layer to get prediction
        if isinstance(self.predictor.horizon, int):
            horizon = self.predictor.horizon
        elif self.predictor.predictor_method == 'rolling':
            horizon = max(self.predictor.horizon)
        elif self.predictor.predictor_method == 'separate':
            horizon = len(self.predictor.horizon)
        else:
            raise ValueError

        output = torch.empty(x.shape[0], horizon, self.predictor.output_size)
        output = output.to(self.predictor.device)
        for i in range(horizon):
            out_i = self.fc(hidden)
            output[:, i, :] = out_i
            zeros_i = torch.zeros_like(out_i)
            in_i = []
            for feature in self.predictor.traffic_data.features:
                if feature in self.predictor.target_feature:
                    in_i.append(out_i)
                else:
                    in_i.append(zeros_i)
            in_i = torch.cat(in_i, dim=1)
            if self.predictor.model_type == 'lstm':
                hidden, cell = self.rnn(in_i, (hidden, cell))
            else:
                hidden = self.rnn(in_i, hidden)

        if isinstance(self.predictor.horizon, list) and self.predictor.predictor_method == 'rolling':
            horizon = [i - 1 for i in self.predictor.horizon]
            output = output[:, horizon, :]

        return output


class GlobalRNNPredictor(BaseRNNPredictor):
    """
    Predictor using RNN with LSTM or GRU cells

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
    """

    def __init__(self, traffic_data, hidden_size, batch_size, seq_len, model_type, max_epoch=5, learning_rate=1e-4,
                 target_feature=None, horizon=1, predictor_method='rolling', save=False, save_path='../models/gru/', **optimizer_args):
        super().__init__(traffic_data=traffic_data, hidden_size=hidden_size, batch_size=batch_size, seq_len=seq_len, model_type=model_type,
                         max_epoch=max_epoch, learning_rate=learning_rate, target_feature=target_feature, horizon=horizon, save=save,
                         save_path=save_path, **optimizer_args)

        self.predictor_method = predictor_method

        # number of input/output features (int)
        self.input_size = len(self.traffic_data.links) * len(self.traffic_data.features)
        self.output_size = len(self.traffic_data.links) * len(self.target_feature)

        # create model
        self.model = GlobalRNNModel(self)
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return "rnn-{}-h{}-s{}-b{}-lr{}".format(self.model_type, self.hidden_size, self.seq_len, self.batch_size, self.learning_rate)

    def train(self):
        return super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)
