import numpy as np
import torch
import torch.nn as nn

from src.prediction_models.neural_nets.rnn.base_rnn import BaseRNNPredictor


class GAGRUModel(nn.Module):
    """
    PyTorch model for GAGRU Predictor. Using a custom GRU implementation

    Parameters
    ----------
    predictor : GAGRUPredictor
        The GAGRUPredictor instance that is the parent of this model

    Attributes
    ----------
    predictor : GAGRUPredictor
        The GAGRUPredictor instance that is the parent of this model
    num_links : int
        The number of links in the dataset
    num_features : int
        The number of features in the dataset
    leaky_relu
        The leaky ReLU layer inside the GAT implementation
    softmax
        The softmax layer inside the GAT implementation
    """

    def __init__(self, predictor):
        super(GAGRUModel, self).__init__()

        self.predictor = predictor

        self.num_links = len(self.predictor.traffic_data.links)
        self.num_features = len(self.predictor.source_feature)
        self.hidden_size = self.predictor.hidden_size

        self.broken_prob = 0

        if self.predictor.attention == 'input' or self.predictor.attention == 'full':
            self.input_attention = True
        else:
            self.input_attention = False

        if self.predictor.attention == 'hidden' or self.predictor.attention == 'full':
            self.hidden_attention = True
        else:
            self.hidden_attention = False

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)

        if self.input_attention:
            self.reset_self_attn_input = nn.Linear(self.predictor.hidden_size, 1, bias=False)
            self.reset_neighbour_attn_input = nn.Linear(self.predictor.hidden_size, 1, bias=False)
        if self.hidden_attention:
            self.reset_self_attn_hidden = nn.Linear(self.predictor.hidden_size, 1, bias=False)
            self.reset_neighbour_attn_hidden = nn.Linear(self.predictor.hidden_size, 1, bias=False)
        self.reset_hidden = nn.Linear(self.predictor.hidden_size, self.predictor.hidden_size)
        self.reset_input = nn.Linear(self.num_features, self.predictor.hidden_size)

        if self.input_attention:
            self.update_self_attn_input = nn.Linear(self.predictor.hidden_size, 1, bias=False)
            self.update_neighbour_attn_input = nn.Linear(self.predictor.hidden_size, 1, bias=False)
        if self.hidden_attention:
            self.update_self_attn_hidden = nn.Linear(self.predictor.hidden_size, 1, bias=False)
            self.update_neighbour_attn_hidden = nn.Linear(self.predictor.hidden_size, 1, bias=False)
        self.update_hidden = nn.Linear(self.predictor.hidden_size, self.predictor.hidden_size)
        self.update_input = nn.Linear(self.num_features, self.predictor.hidden_size)

        if self.input_attention:
            self.new_self_attn_input = nn.Linear(self.predictor.hidden_size, 1, bias=False)
            self.new_neighbour_attn_input = nn.Linear(self.predictor.hidden_size, 1, bias=False)
        if self.hidden_attention:
            self.new_self_attn_hidden = nn.Linear(self.predictor.hidden_size, 1, bias=False)
            self.new_neighbour_attn_hidden = nn.Linear(self.predictor.hidden_size, 1, bias=False)
        self.new_hidden = nn.Linear(self.predictor.hidden_size, self.predictor.hidden_size)
        self.new_input = nn.Linear(self.num_features, self.predictor.hidden_size)

        self.fc = nn.Linear(self.predictor.hidden_size, len(self.predictor.target_feature))

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
        if self.input_attention:
            reset_transformed = self.reset_input(in_i.transpose(1, 2))  # [batch_size, num_links, hidden_size]
            reset_self_attn_input = self.reset_self_attn_input(reset_transformed).repeat(1, 1, self.num_links)
            reset_neighbour_attn_input = self.reset_neighbour_attn_input(reset_transformed).transpose(1, 2).repeat(1, self.num_links, 1)
            reset_attn_input = self.softmax(self.leaky_relu(reset_self_attn_input + reset_neighbour_attn_input) * self.predictor.mask)
            reset_input = self.reset_input(torch.bmm(in_i, reset_attn_input).transpose(1, 2))

            update_transformed = self.update_input(in_i.transpose(1, 2))
            update_self_attn_input = self.update_self_attn_input(update_transformed).repeat(1, 1, self.num_links)
            update_neighbour_attn_input = self.update_neighbour_attn_input(update_transformed).transpose(1, 2).repeat(1, self.num_links, 1)
            update_attn_input = self.softmax(self.leaky_relu(update_self_attn_input + update_neighbour_attn_input) * self.predictor.mask)
            update_input = self.update_input(torch.bmm(in_i, update_attn_input).transpose(1, 2))

            new_transformed = self.new_input(in_i.transpose(1, 2))
            new_self_attn_input = self.new_self_attn_input(new_transformed).repeat(1, 1, self.num_links)
            new_neighbour_attn_input = self.new_neighbour_attn_input(new_transformed).transpose(1, 2).repeat(1, self.num_links, 1)
            new_attn_input = self.softmax(self.leaky_relu(new_self_attn_input + new_neighbour_attn_input) * self.predictor.mask)
            new_input = self.new_input(torch.bmm(in_i, new_attn_input).transpose(1, 2))
        else:
            reset_input = self.reset_input(in_i.transpose(1, 2))
            update_input = self.update_input(in_i.transpose(1, 2))
            new_input = self.new_input(in_i.transpose(1, 2))

        if self.hidden_attention:
            reset_transformed = self.reset_hidden(hidden.transpose(1, 2))  # [batch_size, num_links, hidden_size]
            reset_self_attn_hidden = self.reset_self_attn_hidden(reset_transformed).repeat(1, 1, self.num_links)
            reset_neighbour_attn_hidden = self.reset_neighbour_attn_hidden(reset_transformed).transpose(1, 2).repeat(1, self.num_links, 1)
            reset_attn_hidden = self.softmax(self.leaky_relu(reset_self_attn_hidden + reset_neighbour_attn_hidden) * self.predictor.mask)
            reset_hidden = self.reset_hidden(torch.bmm(hidden, reset_attn_hidden).transpose(1, 2))

            update_transformed = self.update_hidden(hidden.transpose(1, 2))
            update_self_attn_hidden = self.update_self_attn_hidden(update_transformed).repeat(1, 1, self.num_links)
            update_neighbour_attn_hidden = self.update_neighbour_attn_hidden(update_transformed).transpose(1, 2).repeat(1, self.num_links, 1)
            update_attn_hidden = self.softmax(self.leaky_relu(update_self_attn_hidden + update_neighbour_attn_hidden) * self.predictor.mask)
            update_hidden = self.update_hidden(torch.bmm(hidden, update_attn_hidden).transpose(1, 2))

            new_transformed = self.new_hidden(hidden.transpose(1, 2))
            new_self_attn_hidden = self.new_self_attn_hidden(new_transformed).repeat(1, 1, self.num_links)
            new_neighbour_attn_hidden = self.new_neighbour_attn_hidden(new_transformed).transpose(1, 2).repeat(1, self.num_links, 1)
            new_attn_hidden = self.softmax(self.leaky_relu(new_self_attn_hidden + new_neighbour_attn_hidden) * self.predictor.mask)
            new_hidden = self.new_hidden(torch.bmm(hidden, new_attn_hidden).transpose(1, 2))
        else:
            reset_hidden = self.reset_hidden(hidden.transpose(1, 2))
            update_hidden = self.update_hidden(hidden.transpose(1, 2))
            new_hidden = self.new_hidden(hidden.transpose(1, 2))

        reset = torch.sigmoid(reset_hidden + reset_input)
        update = torch.sigmoid(update_hidden + update_input)
        new = torch.tanh(reset * new_hidden + new_input)
        hidden = (1 - update) * new + update * hidden.transpose(1, 2)

        hidden = hidden.transpose(1, 2)  # [batch_size, hidden_size, num_links]

        return hidden

    def forward(self, x):
        # Set an initial hidden state
        hidden = torch.randn(x.shape[0], self.predictor.hidden_size, self.num_links)  # [batch_size, hidden_size, num_links]
        hidden = hidden.to(self.predictor.device)

        # Forward propagate the GRNN
        for i in range(self.predictor.seq_len):
            # input at time i
            in_i = x[:, i, :].view((x.shape[0], self.num_features, self.num_links))  # [batch_size, num_features, num_links]

            hidden = self.forward_propagate(hidden, in_i)

        # the length of prediction horizon
        horizon = self.horizon_length()

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

            hidden = self.forward_propagate(hidden, in_i)

        if isinstance(self.predictor.horizon, list) and self.predictor.predictor_method == 'rolling':
            horizon = [i - 1 for i in self.predictor.horizon]
            output = output[:, horizon, :]

        return output


class GAGRUPredictor(BaseRNNPredictor):
    """
    Graph Attention Network (Velickovic, P. at al., 2018) combined with the GRU by replacing the GRU matrix multiplication with the
    GAT convolution operation

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
    attention : {'input', 'hidden', 'full'}
        Where to apply the GAT convolution operation within the GRU
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
        The GAGRU PyTorch model
    """

    def __init__(self, traffic_data, hidden_size, batch_size, seq_len, attention='input', conv_radius=5, max_epoch=5, learning_rate=1e-4,
                 target_feature=None, horizon=1, predictor_method='rolling', save=False, save_path='../models/gagru/', **optimizer_args):
        super().__init__(traffic_data=traffic_data, hidden_size=hidden_size, batch_size=batch_size, seq_len=seq_len, model_type='gru',
                         max_epoch=max_epoch, learning_rate=learning_rate, target_feature=target_feature, horizon=horizon, save=save,
                         save_path=save_path, **optimizer_args)

        self.predictor_method = predictor_method
        self.attention = attention
        self.conv_radius = conv_radius

        self.output_size = len(self.traffic_data.links) * len(self.target_feature)

        # create model and send to gpu
        self.model = GAGRUModel(self)
        self.model.to(self.device)

        # build gradient mask to limit propagation
        self.mask = self.build_mask()

        # save the hyperparameters before training
        if self.save_model:
            self.save(self.save_path)

    def __str__(self):
        return f"ga_gru-att-{self.attention}-{self.predictor_method}-h{self.hidden_size}-c{self.conv_radius}-s{self.seq_len}-b{self.batch_size}-" \
               f"lr{self.learning_rate}-w{self.optimizer_args['weight_decay']}"

    def train(self):
        super().train()

    def evaluate(self, mode, links=None):
        return super().evaluate(mode, links=links)

    # def build_adj_matrix(self):
    #     """
    #     Build the modified adjacency matrix of the network to facilitate hidden state propagation of the GRNN
    #     The modified adjacency matrix has 1's on the diagonal (every link always propagates to itself undiminished), alpha on [i, j] if there's a
    #     connection between link i and link j, and 0's everywhere else.
    #
    #     Returns
    #     -------
    #     adj_matrix
    #         The modified adjacency matrix
    #     """
    #     num_links = len(self.traffic_data.links)
    #     adj_matrix = torch.eye(num_links, device=self.device)
    #     for i, link in enumerate(self.traffic_data.links):
    #         neighbors = []
    #         for direction in ['left', 'straight', 'right']:
    #             neighbors.extend(self.traffic_data.network.downstream[link][direction])
    #             neighbors.extend(self.traffic_data.network.upstream[link][direction])
    #         link_indices = np.where(np.isin(self.traffic_data.links, neighbors))[0]
    #         adj_matrix[i, link_indices] = self.alpha
    #
    #     return adj_matrix

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
        num_links = len(self.traffic_data.links)
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
