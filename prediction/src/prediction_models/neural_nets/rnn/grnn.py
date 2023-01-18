import numpy as np
import torch
import torch.nn as nn

from src.prediction_models.neural_nets.rnn.base_rnn import BaseRNNPredictor


class GRNNModel(nn.Module):
    """
    PyTorch model for GRNN Predictor. Using a custom GRU implementation

    Parameters
    ----------
    predictor : GRNNPredictor
        The GRNNPredictor instance that is the parent of this model

    Attributes
    ----------
    predictor : GRNNPredictor
        The GRNNPredictor instance that is the parent of this model
    num_links : int
        The number of links in the dataset
    num_features : int
        The number of features in the dataset
    adj_matrix
        The adjacency matrix of the road network graph, with the diagonal set to 1 and neighbors set to alpha
    reset_hidden
        The U matrix in the reset gate of the GRU implementation
    reset_input
        The W matrix in the reset gate of the GRU implementation
    update_hidden
        The U matrix in the update gate of the GRU implementation
    update_input
        The W matrix in the update gate of the GRU implementation
    new_hidden
        The U matrix in the output unit of the GRU implementation
    new_input
        The W matrix in the output unit of the GRU implementation
    fc
        The fully connected layer used to generate output from hidden representations
    """
    def __init__(self, predictor):
        super(GRNNModel, self).__init__()

        self.predictor = predictor

        self.num_links = len(self.predictor.traffic_data.links)
        self.num_features = len(self.predictor.source_feature)
        self.adj_matrix = self.predictor.build_adj_matrix()

        self.reset_hidden = nn.Linear(self.predictor.hidden_size, self.predictor.hidden_size)
        self.reset_input = nn.Linear(self.num_features, self.predictor.hidden_size)

        self.update_hidden = nn.Linear(self.predictor.hidden_size, self.predictor.hidden_size)
        self.update_input = nn.Linear(self.num_features, self.predictor.hidden_size)

        self.new_hidden = nn.Linear(self.predictor.hidden_size, self.predictor.hidden_size)
        self.new_input = nn.Linear(self.num_features, self.predictor.hidden_size)

        self.fc = nn.Linear(self.predictor.hidden_size, len(self.predictor.target_feature))

    def forward(self, x):
        # Set an initial hidden state
        hidden = torch.randn(x.shape[0], self.predictor.hidden_size, self.num_links)
        hidden = hidden.to(self.predictor.device)

        # Forward propagate the GRNN
        for i in range(self.predictor.seq_len):
            # input at time i
            in_i = x[:, i, :].view((x.shape[0], self.num_features, self.num_links))  # [batch_size, num_features, num_links]
            in_i = in_i.transpose(1, 2)  # [batch_size, num_links, num_features]

            # propagate hidden representations to neighbors using the adjacency matrix
            hidden = torch.matmul(hidden, self.adj_matrix)  # [batch_size, hidden_size, num_links]
            hidden = hidden.transpose(1, 2)  # [batch_size, num_links, hidden_size]

            # update the hidden representations using GRU update rules
            reset = torch.sigmoid(self.reset_hidden(hidden) + self.reset_input(in_i))
            update = torch.sigmoid(self.update_hidden(hidden) + self.update_input(in_i))
            new = torch.tanh(reset * self.new_hidden(hidden) + self.new_input(in_i))
            hidden = (1 - update) * new + update * hidden
            hidden = hidden.transpose(1, 2)  # [batch_size, hidden_size, num_links]

        # the length of prediction horizon
        if isinstance(self.predictor.horizon, int):
            horizon = self.predictor.horizon
        elif self.predictor.predictor_method == 'rolling':
            horizon = max(self.predictor.horizon)
        elif self.predictor.predictor_method == 'separate':
            horizon = len(self.predictor.horizon)
        else:
            raise ValueError

        # create an empty tensor for the output
        output = torch.empty(x.shape[0], horizon, self.predictor.output_size)
        output = output.to(self.predictor.device)

        # unroll the prediction horizon
        for i in range(horizon):
            # Pass the hidden representations to the fully connected layer to get prediction
            out_i = self.fc(hidden.transpose(1, 2))  # [batch_size, num_links, num_out_features]
            out_i = out_i.transpose(1, 2)  # [batch_size, num_out_features, num_links]
            output[:, i, :] = out_i.view((x.shape[0], self.predictor.output_size))

            # the input to the next time step is the current prediction
            # prediction does not necessarily contain all input features, so the non-predicted features are masked to 0
            zeros_i = torch.zeros_like(out_i)
            in_i = []
            for feature in self.predictor.source_feature:
                if feature in self.predictor.target_feature:
                    in_i.append(out_i)
                else:
                    in_i.append(zeros_i)
            in_i = torch.cat(in_i, dim=1)  # [batch_size, num_features, num_links]
            in_i = in_i.transpose(1, 2)  # [batch_size, num_links, num_features]

            # propagate hidden representations to neighbors using the adjacency matrix
            hidden = torch.matmul(hidden, self.adj_matrix)  # [batch_size, hidden_size, num_links]
            hidden = hidden.transpose(1, 2)  # [batch_size, num_links, hidden_size]

            # update the hidden representations using GRU update rules
            reset = torch.sigmoid(self.reset_hidden(hidden) + self.reset_input(in_i))
            update = torch.sigmoid(self.update_hidden(hidden) + self.update_input(in_i))
            new = torch.tanh(reset * self.new_hidden(hidden) + self.new_input(in_i))
            hidden = (1 - update) * new + update * hidden
            hidden = hidden.transpose(1, 2)  # [batch_size, hidden_size, num_links]

        if isinstance(self.predictor.horizon, list) and self.predictor.predictor_method == 'rolling':
            horizon = [i - 1 for i in self.predictor.horizon]
            output = output[:, horizon, :]

        return output


class GRNNPredictor(BaseRNNPredictor):
    """
    Gated Recurrent Neural Network (Wang, X. et al., 2018)

    Parameters
    ----------
    traffic_data : TrafficData
        Traffic data to load
    hidden_size : int
        The size of hidden representation of each node
    batch_size : int
        The size of each training batch
    seq_len : int
        The length of each sequence
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

    Attributes
    ----------
    alpha : float
        The alpha parameter for neighbor propagation in the GRNN framework
    predictor_method : {'rolling', 'separate'}
    output_size : int
        The number of output features
    model
        The GRNN PyTorch model
    """
    def __init__(self, traffic_data, hidden_size, batch_size, seq_len, alpha=0.2, max_epoch=5, learning_rate=1e-4, target_feature=None, horizon=1,
                 predictor_method='rolling', save=False, save_path='../models/grnn/', **optimizer_args):
        super().__init__(traffic_data=traffic_data, hidden_size=hidden_size, batch_size=batch_size, seq_len=seq_len, model_type=None,
                         max_epoch=max_epoch, learning_rate=learning_rate, target_feature=target_feature, horizon=horizon, save=save,
                         save_path=save_path, **optimizer_args)

        self.alpha = alpha
        self.predictor_method = predictor_method

        self.output_size = len(self.traffic_data.links) * len(self.target_feature)

        # create model and send to gpu
        self.model = GRNNModel(self)
        self.model.to(self.device)

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return "grnn-h{}-s{}-b{}-lr{}-a{}".format(self.hidden_size, self.seq_len, self.batch_size, self.learning_rate, self.alpha)

    def train(self):
        super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)

    def build_adj_matrix(self):
        """
        Build the modified adjacency matrix of the network to facilitate hidden state propagation of the GRNN
        The modified adjacency matrix has 1's on the diagonal (every link always propagates to itself undiminished), alpha on [i, j] if there's a
        connection between link i and link j, and 0's everywhere else.

        Returns
        -------
        adj_matrix
            The modified adjacency matrix
        """
        num_links = len(self.traffic_data.links)
        adj_matrix = torch.eye(num_links, device=self.device)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = []
            for direction in ['left', 'straight', 'right']:
                neighbors.extend(self.traffic_data.network.downstream[link][direction])
                neighbors.extend(self.traffic_data.network.upstream[link][direction])
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            adj_matrix[i, link_indices] = self.alpha
        return adj_matrix
