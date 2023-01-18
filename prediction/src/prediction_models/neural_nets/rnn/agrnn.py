import math

import numpy as np
import torch
import torch.nn as nn

from src.prediction_models.neural_nets.rnn.base_rnn import BaseRNNPredictor


class AGRNNCell(nn.Module):
    """
    Attributes
    ----------
    reset_hidden_weight
        The hidden-hidden weights in the reset gate of the GRU implementation
    reset_hidden_bias
        The hidden-hidden bias in the reset gate of the GRU implementation
    reset_input_weight
        The input-hidden weights in the reset gate of the GRU implementation
    reset_input_bias
        The input-hidden bias in the reset gate of the GRU implementation
    update_hidden_weight
        The hidden-hidden weights in the update gate of the GRU implementation
    update_hidden_bias
        The hidden-hidden bias in the update gate of the GRU implementation
    update_input_weight
        The input-hidden weights in the update gate of the GRU implementation
    update_input_bias
        The input-hidden bias in the update gate of the GRU implementation
    new_hidden_weight
        The hidden-hidden weights in the output unit of the GRU implementation
    new_hidden_bias
        The hidden-hidden bias in the output unit of the GRU implementation
    new_input_weight
        The input-hidden weights in the output unit of the GRU implementation
    new_input_bias
        The input-hidden bias in the output unit of the GRU implementation
    """

    def __init__(self, num_links, num_features, hidden_size):
        super(AGRNNCell, self).__init__()

        self.hidden_size = hidden_size

        self.reset_hidden_weight = nn.Parameter(torch.empty(num_links, hidden_size, hidden_size), requires_grad=True)
        self.reset_hidden_bias = nn.Parameter(torch.empty(hidden_size, num_links), requires_grad=True)
        self.reset_input_weight = nn.Parameter(torch.empty(num_links, num_features, hidden_size), requires_grad=True)
        self.reset_input_bias = nn.Parameter(torch.empty(hidden_size, num_links), requires_grad=True)

        self.update_hidden_weight = nn.Parameter(torch.empty(num_links, hidden_size, hidden_size), requires_grad=True)
        self.update_hidden_bias = nn.Parameter(torch.empty(hidden_size, num_links), requires_grad=True)
        self.update_input_weight = nn.Parameter(torch.empty(num_links, num_features, hidden_size), requires_grad=True)
        self.update_input_bias = nn.Parameter(torch.empty(hidden_size, num_links), requires_grad=True)

        self.new_hidden_weight = nn.Parameter(torch.empty(num_links, hidden_size, hidden_size), requires_grad=True)
        self.new_hidden_bias = nn.Parameter(torch.empty(hidden_size, num_links), requires_grad=True)
        self.new_input_weight = nn.Parameter(torch.empty(num_links, num_features, hidden_size), requires_grad=True)
        self.new_input_bias = nn.Parameter(torch.empty(hidden_size, num_links), requires_grad=True)

    def initialize(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.reset_hidden_weight, self.reset_hidden_bias, self.reset_input_weight, self.reset_input_bias,
                       self.update_hidden_weight, self.update_hidden_bias, self.update_input_weight, self.update_input_bias,
                       self.new_hidden_weight, self.new_hidden_bias, self.new_input_weight, self.new_input_bias]:
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hidden, in_i):
        """
        Forward propagate the RNN by a single time step

        Parameters
        ----------
        hidden
            The previous hidden state (shape [batch_size, hidden_size, num_links])
        in_i
            The current input (shape [batch_size, num_features, num_links])

        Returns
        -------
        hidden
            The updated hidden state (shape [batch_size, hidden_size, num_links])
        """
        reset = torch.sigmoid(torch.einsum('bij, jik->bkj', hidden, self.reset_hidden_weight) + self.reset_hidden_bias
                              + torch.einsum('bij, jik->bkj', in_i, self.reset_input_weight) + self.reset_input_bias)
        update = torch.sigmoid(torch.einsum('bij, jik->bkj', hidden, self.update_hidden_weight) + self.update_hidden_bias
                               + torch.einsum('bij, jik->bkj', in_i, self.update_input_weight) + self.update_input_bias)
        new = torch.tanh(reset * (torch.einsum('bij, jik->bkj', hidden, self.new_hidden_weight) + self.new_hidden_bias)
                         + torch.einsum('bij, jik->bkj', in_i, self.new_input_weight) + self.new_input_bias)
        hidden = (1 - update) * new + update * hidden

        return hidden


class AGRNNModel(nn.Module):
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
    hidden_attention
        The attention matrix of the road network graph, with the diagonal initialized to 1 and immediate neighbors initialized to alpha
    reset_hidden_weight
        The hidden-hidden weights in the reset gate of the GRU implementation
    reset_hidden_bias
        The hidden-hidden bias in the reset gate of the GRU implementation
    reset_input_weight
        The input-hidden weights in the reset gate of the GRU implementation
    reset_input_bias
        The input-hidden bias in the reset gate of the GRU implementation
    update_hidden_weight
        The hidden-hidden weights in the update gate of the GRU implementation
    update_hidden_bias
        The hidden-hidden bias in the update gate of the GRU implementation
    update_input_weight
        The input-hidden weights in the update gate of the GRU implementation
    update_input_bias
        The input-hidden bias in the update gate of the GRU implementation
    new_hidden_weight
        The hidden-hidden weights in the output unit of the GRU implementation
    new_hidden_bias
        The hidden-hidden bias in the output unit of the GRU implementation
    new_input_weight
        The input-hidden weights in the output unit of the GRU implementation
    new_input_bias
        The input-hidden bias in the output unit of the GRU implementation
    fc_weight
        The fully connected layer weight used to generate output from hidden representations
    fc_bias
        The fully connected layer bias used to generate output from hidden representations
    """

    def __init__(self, predictor):
        super(AGRNNModel, self).__init__()

        self.predictor = predictor

        if self.predictor.attention == 'input' or self.predictor.attention == 'full':
            self.use_input_attention = True
        else:
            self.use_input_attention = False

        if self.predictor.attention == 'hidden' or self.predictor.attention == 'full':
            self.use_hidden_attention = True
        else:
            self.use_hidden_attention = False

        if not self.use_input_attention and not self.use_hidden_attention:
            raise ValueError

        self.num_links = self.predictor.traffic_data.data.shape[1]  # (F, N, T) -> N
        self.num_features = len(self.predictor.source_feature)
        self.hidden_size = self.predictor.hidden_size

        if self.use_input_attention:
            self.input_attention = nn.Linear(self.num_links, self.num_links, bias=False)
            self.input_attention.weight.data = 0.01 * (torch.rand(self.num_links, self.num_links) - 0.5)
        if self.use_hidden_attention:
            self.hidden_attention = nn.Linear(self.num_links, self.num_links, bias=False)
            self.hidden_attention.weight.data = 0.01 * (torch.rand(self.num_links, self.num_links) - 0.5)

        self.broken_prob = 0

        # self.encoder = AGRNNCell(self.num_links, self.num_features, self.hidden_size)
        # if self.predictor.decoder:
        #     self.decoder = AGRNNCell(self.num_links, self.num_features, self.hidden_size)

        self.reset_hidden_weight = nn.Parameter(torch.empty(self.num_links, self.hidden_size, self.hidden_size), requires_grad=True)
        self.reset_hidden_bias = nn.Parameter(torch.empty(self.hidden_size, self.num_links), requires_grad=True)
        self.reset_input_weight = nn.Parameter(torch.empty(self.num_links, self.num_features, self.hidden_size), requires_grad=True)
        self.reset_input_bias = nn.Parameter(torch.empty(self.hidden_size, self.num_links), requires_grad=True)

        self.update_hidden_weight = nn.Parameter(torch.empty(self.num_links, self.hidden_size, self.hidden_size), requires_grad=True)
        self.update_hidden_bias = nn.Parameter(torch.empty(self.hidden_size, self.num_links), requires_grad=True)
        self.update_input_weight = nn.Parameter(torch.empty(self.num_links, self.num_features, self.hidden_size), requires_grad=True)
        self.update_input_bias = nn.Parameter(torch.empty(self.hidden_size, self.num_links), requires_grad=True)

        self.new_hidden_weight = nn.Parameter(torch.empty(self.num_links, self.hidden_size, self.hidden_size), requires_grad=True)
        self.new_hidden_bias = nn.Parameter(torch.empty(self.hidden_size, self.num_links), requires_grad=True)
        self.new_input_weight = nn.Parameter(torch.empty(self.num_links, self.num_features, self.hidden_size), requires_grad=True)
        self.new_input_bias = nn.Parameter(torch.empty(self.hidden_size, self.num_links), requires_grad=True)

        self.fc_weight = nn.Parameter(torch.empty(self.num_links, self.hidden_size, len(self.predictor.target_feature)), requires_grad=True)
        self.fc_bias = nn.Parameter(torch.empty(len(self.predictor.target_feature), self.num_links), requires_grad=True)

        self.initialize()

    def initialize(self):
        """
        Initializes the network weights using the same initialization scheme as PyTorch GRU
        https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.reset_hidden_weight, self.reset_hidden_bias, self.reset_input_weight, self.reset_input_bias,
                       self.update_hidden_weight, self.update_hidden_bias, self.update_input_weight, self.update_input_bias,
                       self.new_hidden_weight, self.new_hidden_bias, self.new_input_weight, self.new_input_bias, self.fc_weight, self.fc_bias]:
            nn.init.uniform_(weight, -stdv, stdv)

    def horizon_length(self):
        """
        Get the length of prediction horizon

        Returns
        -------
        horizon
            the length of prediction horizon
        """
        if isinstance(self.predictor.horizon, int):
            horizon = self.predictor.horizon
        elif self.predictor.predictor_method == 'rolling':
            horizon = max(self.predictor.horizon)
        elif self.predictor.predictor_method == 'separate':
            horizon = len(self.predictor.horizon)
        else:
            raise ValueError

        return horizon

    def forward_propagate(self, hidden, in_i):
        """
        Forward propagate the RNN by a single time step

        Parameters
        ----------
        hidden
            The previous hidden state (shape [batch_size, hidden_size, num_links])
        in_i
            The current input (shape [batch_size, num_features, num_links])

        Returns
        -------
        hidden
            The updated hidden state (shape [batch_size, hidden_size, num_links])
        """
        reset = torch.sigmoid(torch.einsum('bij, jik->bkj', hidden, self.reset_hidden_weight) + self.reset_hidden_bias
                              + torch.einsum('bij, jik->bkj', in_i, self.reset_input_weight) + self.reset_input_bias)
        update = torch.sigmoid(torch.einsum('bij, jik->bkj', hidden, self.update_hidden_weight) + self.update_hidden_bias
                               + torch.einsum('bij, jik->bkj', in_i, self.update_input_weight) + self.update_input_bias)
        new = torch.tanh(reset * (torch.einsum('bij, jik->bkj', hidden, self.new_hidden_weight) + self.new_hidden_bias)
                         + torch.einsum('bij, jik->bkj', in_i, self.new_input_weight) + self.new_input_bias)
        hidden = (1 - update) * new + update * hidden

        return hidden

    def forward(self, x):
        # Set an initial hidden state
        hidden = torch.zeros(x.shape[0], self.predictor.hidden_size, self.num_links)  # [batch_size, hidden_size, num_links]
        hidden = hidden.to(self.predictor.device)

        # Forward propagate the GRNN
        for i in range(self.predictor.seq_len):
            # input at time i
            in_i = x[:, i, :].view((x.shape[0], self.num_features, self.num_links))  # [batch_size, num_features, num_links]

            # propagate to neighbors using the attention matrix
            if self.use_input_attention:
                in_i = self.input_attention(in_i)  # [batch_size, hidden_size, num_links]
            if self.use_hidden_attention:
                hidden = self.hidden_attention(hidden)  # [batch_size, hidden_size, num_links]

            hidden = self.forward_propagate(hidden, in_i)

        # the length of prediction horizon
        horizon = self.horizon_length()

        # create an empty tensor for the output
        output = torch.empty(x.shape[0], horizon, self.predictor.output_size)
        output = output.to(self.predictor.device)

        # unroll the prediction horizon
        for i in range(horizon):
            # Pass the hidden representations to the fully connected layer to get prediction
            out_i = torch.einsum('bij, jik->bkj', hidden, self.fc_weight) + self.fc_bias  # [batch_size, num_out_features, num_links]
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

            # propagate to neighbors using the attention matrix
            if self.use_input_attention:
                in_i = self.input_attention(in_i)  # [batch_size, hidden_size, num_links]
            if self.use_hidden_attention:
                hidden = self.hidden_attention(hidden)  # [batch_size, hidden_size, num_links]

            hidden = self.forward_propagate(hidden, in_i)

        if isinstance(self.predictor.horizon, list) and self.predictor.predictor_method == 'rolling':
            horizon = [i - 1 for i in self.predictor.horizon]
            output = output[:, horizon, :]

        return output


class AGRNNPredictor(BaseRNNPredictor):
    """
    Gated Recurrent Neural Network (Wang, X. et al., 2018) with attention, allowing propagation to more distant links

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
        The AGRNN PyTorch model
    """

    def __init__(self, traffic_data, hidden_size, batch_size, seq_len, is_pems, adj_type='connect', attention='input', decoder=False, conv_radius=5,
                 alpha=0.2, max_epoch=5, learning_rate=1e-4, source_feature=None, target_feature=None, horizon=1, predictor_method='rolling',
                 save=False, save_interval=50, save_path='../models/agrnn/', **optimizer_args):
        super().__init__(traffic_data=traffic_data, hidden_size=hidden_size, batch_size=batch_size, seq_len=seq_len, model_type=None,
                         max_epoch=max_epoch, learning_rate=learning_rate, source_feature=source_feature, target_feature=target_feature,
                         horizon=horizon, save=save, save_path=save_path, save_interval=save_interval, is_pems=is_pems, **optimizer_args)
        self.adj_type = adj_type
        self.attention = attention
        self.decoder = decoder
        self.alpha = alpha
        self.conv_radius = conv_radius
        self.predictor_method = predictor_method
        self.num_links = traffic_data.data.shape[1]  # (F, N, T) -> N
        self.output_size = self.num_links * len(self.target_feature)
        # create model and send to gpu
        self.model = AGRNNModel(self)
        self.model.to(self.device)

        # build gradient mask to limit propagation
        self.mask = self.build_mask()

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return f"agrnn-{self.attention}-{self.predictor_method}-h{self.hidden_size}-s{self.seq_len}-hor{self.horizon}-b{self.batch_size}-" \
               f"lr{self.learning_rate}-w{self.optimizer_args['weight_decay']}"

    def train(self):
        """
        Trains the neural network with MSE Loss and Adam optimizer
        """
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
        # if self.is_pems:
        #     if self.adj_type == "connect":
        #         adj_matrix = np.eye(self.traffic_data.data.shape[1]) + self.alpha * self.traffic_data.adjacency_matrix[self.adj_type]
        #     else:
        #         adj_matrix = self.traffic_data.adjacency_matrix[self.adj_type]
        #     return torch.from_numpy(adj_matrix).float().to(self.device)
        #
        # num_links = len(self.traffic_data.links)
        adj_matrix = torch.eye(self.num_links, device=self.device)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = []
            for direction in ['left', 'straight', 'right']:
                neighbors.extend(self.traffic_data.network.downstream[link][direction])
                neighbors.extend(self.traffic_data.network.upstream[link][direction])
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            adj_matrix[i, link_indices] = self.alpha

        return adj_matrix

    def build_mask(self):
        """
        Build a mask matrix that represents the neighborhood for each link, [i, j] = 1 if link j is in link i's neighborhood, otherwise [i, j] = 0.
        This gradient to the attention matrix is multiplied by this mask during backprop so that the attention value remains 0 for links that are
        far apart.

        Returns
        -------
        mask
            The mask matrix

        """
        num_links = self.num_links
        mask = torch.eye(num_links, device=self.device)
        for i, link in enumerate(self.traffic_data.links):
            neighbors = self.get_neighbors(link)
            link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
            mask[i, link_indices] = 1

        return mask

    def get_neighbors(self, target_link):
        """
        Get all the links within n hops from target by breadth first search with n iterations

        Parameters
        ----------
        target_link : str
            The target link to start the search from
        n : int
            Number of iterations of breadth first search

        Returns
        -------
        list of str
            The links within the neighborhood of target, excluding target itself
        """
        neighbors = [target_link]
        # breadth first search with n iterations
        for _ in range(self.conv_radius):
            new_neighbors = [n for n in neighbors]
            for neighbor in neighbors:
                for direction in ['left', 'straight', 'right']:
                    new_neighbors.extend(self.traffic_data.network.downstream[neighbor][direction])
                    new_neighbors.extend(self.traffic_data.network.upstream[neighbor][direction])
            new_neighbors = list(set(new_neighbors))
            neighbors = new_neighbors
        # remove target_link itself from neighbors
        neighbors.remove(target_link)
        return neighbors
