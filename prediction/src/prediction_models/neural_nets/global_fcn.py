import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prediction_models.neural_nets.nn_predictor import NNPredictor


class GlobalFCNModel(nn.Module):
    """
    input_size : int
        the input size of the fully connected neural network
    output_size : int
        the output size of the fully connected neural network
    """

    def __init__(self, predictor):
        super(GlobalFCNModel, self).__init__()

        self.predictor = predictor

        # set the activation function
        if self.predictor.activation == 'relu':
            self.activation = F.relu
        elif self.predictor.activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif self.predictor.activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError

        # number of input/output features (int)
        self.input_size = len(self.predictor.traffic_data.links) * len(self.predictor.traffic_data.features) * self.predictor.seq_len
        if isinstance(self.predictor.horizon, int):
            self.output_size = len(self.predictor.traffic_data.links) * len(self.predictor.target_feature) * self.predictor.horizon
        elif isinstance(self.predictor.horizon, list):
            self.output_size = len(self.predictor.traffic_data.links) * len(self.predictor.target_feature) * len(self.predictor.horizon)
        else:
            raise ValueError

        self.fc1 = nn.Linear(self.input_size, self.predictor.hidden_size)
        self.dropout = nn.Dropout(p=self.predictor.dropout)
        self.fc2 = nn.Linear(self.predictor.hidden_size, self.output_size)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        horizon = self.predictor.horizon if isinstance(self.predictor.horizon, int) else len(self.predictor.horizon)
        x = x.view(x.shape[0], horizon, len(self.predictor.traffic_data.links) * len(self.predictor.target_feature))

        return x


class GlobalFCNPredictor(NNPredictor):
    """
    2-layer fully connected neural network with ReLU activation
    Uses data from entire network from the current and past time steps as input features to predict next time step

    Parameters
    ----------
    traffic_data : TrafficData
        Traffic data to load
    hidden_size : int
        The size of hidden layer
    batch_size : int
        The size of each training batch
    max_epoch : int
        when to stop training
    learning_rate : float
    target_feature : list of str
        what features to predict, default: ['speed']
    past_steps : int
        how much past data to include as feature
    save : bool
        whether to save the model
    save_path : str
        where to save the model

    Attributes
    ----------
    hidden_size : int
        The size of hidden layer
    batch_size : int
        The size of each training batch
    max_epoch : int
        number of training epochs
    learning_rate : float
    seq_len : int
        how many timestamps to include as feature
    save_model : bool
        whether to save the model
    save_path : str
        where to save the model
    model
        the PyTorch neural network model
    """

    def __init__(self, traffic_data, hidden_size, batch_size, max_epoch=5, learning_rate=1e-4, past_steps=0, horizon=1, dropout=0, activation='relu',
                 target_feature=None, save=False, save_path='../models/fcn/', **optimizer_args):
        super().__init__(traffic_data, batch_size=batch_size, seq_len=past_steps + 1, max_epoch=max_epoch, learning_rate=learning_rate,
                         target_feature=target_feature, horizon=horizon, save=save, save_path=save_path, **optimizer_args)

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.activation = activation

        # create model
        self.model = GlobalFCNModel(self)
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            self.save(save_path)

    def __str__(self):
        return "fcn-h{}-b{}-lr{}-p{}".format(self.hidden_size, self.batch_size, self.learning_rate, self.seq_len - 1)

    def train(self):
        super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)
