from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf import RecurrentNetwork
from ray.rllib.models.tf.misc import normc_initializer, flatten
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_tf, try_import_torch
from ray.rllib.models.torch.misc import normc_initializer as th_normc_initializer, SlimFC
from tensorflow.keras.layers import Reshape
import numpy as np
from wolf.world.environments.wolfenv.agents.connectors.observation.tdtse import TDTSEStateSpace

tf = try_import_tf()[1]
torch, nn = try_import_torch()


class TdtseModelConfig:
    def __init__(self, obs_space, model_config):
        self.use_progression = bool(model_config["custom_model_config"].get("use_progression", False))
        self.n_nodes = obs_space["tdtse"].shape[0]
        self.n_lanes = obs_space["tdtse"].shape[1]
        self.n_sequence = obs_space["tdtse"].shape[2]
        self.n_channel = obs_space["tdtse"].shape[3]
        # self.extract_features_independantly = bool(
        #     model_config["custom_model_config"].get("extract_features_independantly", False))
        # if self.n_nodes <= 1 and self.extract_features_independantly:
        #     raise Exception("extract_features_independantly makes no sense with n_nodes <= 1")
        self.dense_layer_size_by_node = model_config["custom_model_config"].get("dense_layer_size_by_node", 128)


class TdtseCnnConfig(TdtseModelConfig):
    def __init__(self, obs_space, model_config):
        TdtseModelConfig.__init__(self, obs_space, model_config)
        # self.use_conv1d = bool(model_config["custom_model_config"].get("use_conv1D", False))
        self.filters_size = model_config["custom_model_config"].get("filters_size", 32)


class TdtseRnnConfig(TdtseModelConfig):
    def __init__(self, obs_space, model_config):
        TdtseModelConfig.__init__(self, obs_space, model_config)
        if self.n_sequence > 1:
            raise Exception("TDTSE times axis should be of dimension 1 with RNN models, but is {}"
                            .format(self.n_sequence))
        self.cell_size = model_config["custom_model_config"].get("lstm_cell_size", 128)


class TdtseRnnTfModel(RecurrentNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        raise Exception("this stuff is WIP, no tested nor anything")
        if self.extract_features_independantly:
            raise Exception("TODO")
        RecurrentTFModelV2.__init__(obs_space, action_space, num_outputs, model_config, name)
        if not isinstance(obs_space, TDTSEStateSpace):
            obs_space = obs_space.original_space
            if not isinstance(obs_space, TDTSEStateSpace):
                raise Exception("obs nor obs.original_obs_space is of TDTSEStateSpace")

        config = TdtseRnnConfig(obs_space, model_config)
        self.env_config = config

        self.tdtse = tf.keras.layers.Input(
            shape=obs_space["tdtse"].shape,
            name="tdtse")

        if config.use_progression:
            self.min_progression = tf.keras.layers.Input(
                shape=obs_space["min_progression"].shape,
                name="min_progression")
            self.max_progression = tf.keras.layers.Input(
                shape=obs_space["max_progression"].shape,
                name="max_progression")

        # Define input layers

        if not config.extract_features_independantly:
            hidden_input_shape = (config.cell_size,)
            fcn_layer_size = config.dense_layer_size_by_node * config.n_nodes
        else:
            hidden_input_shape = (config.n_nodes, config.cell_size)
            fcn_layer_size = config.dense_layer_size_by_node

        pre_processor = tf.keras.layers.Dense(
            units=fcn_layer_size,
            name="post_processor",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))

        state_in_h = tf.keras.layers.Input(
            name="state_in_h",
            shape=hidden_input_shape)

        gru = tf.keras.layers.CuDNNGRU(
            units=self.cell_size,
            return_sequences=True,
            return_state=True,
            time_major=True,  # TODO why did we put this stuff? the transpose wont work otherwise but wtf
            name="gru")

        post_processor = tf.keras.layers.Dense(
            units=fcn_layer_size,
            name="post_processor",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))

        # if config.extract_features_independantly:
        # the node layer will be squeezed by the tf.split so we don't need this dimension for the reshape.
        # config.n_sequence should be one
        reshape_layer = Reshape(
            target_shape=((config.n_sequence * config.n_lanes * config.n_channel)))
        # else:
        #     reshape_layer = Reshape(
        #         target_shape=((config.n_sequence * config.n_nodes * config.n_lanes * config.n_channel)))

        x = self.tdtse

        # if not config.extract_features_independantly:
        #     x = reshape_layer(x)
        #     x = pre_processor(x)
        #     x, state_h = gru(inputs=x, initial_state=[state_in_h])
        #     next_hiddens = [state_h]
        # else:
        nodes_features = []
        next_hiddens = []
        # extract tdtse for each agent
        xs = tf.split(self.tdtse, self.tdtse.shape[1], axis=1)
        hiddens = tf.split(state_in_h, self.tdtse.shape[1], axis=1)
        for x, hidden in zip(xs, hiddens):  # apply rnn for each tdtse agent
            x = reshape_layer(x)
            x = pre_processor(x)
            x, state_h = gru(inputs=x, initial_state=[hidden])
            next_hiddens.append(state_h)
            nodes_features.append(x)
        if config.n_nodes > 1:
            x = tf.keras.layers.Concatenate(axis=-1)(nodes_features)
        else:
            x = nodes_features[0]

        x = post_processor(x)

        if config.use_progression:
            x = tf.keras.layers.Concatenate(axis=-1)([x, self.min_progression, self.max_progression])
            inputs = [self.tdtse, self.min_progression, self.max_progression]
        else:
            inputs = [self.tdtse]

        # Create the RNN model
        self.model = tf.keras.Model(
            inputs=inputs,
            outputs=[x, *next_hiddens])

    def forward_rnn(self, input_dict, hidden_state, seq_lens):
        # TODO find a way to get all hidden_state of the same gru cell for all intersections
        model_out, h = self.model([input_dict["obs"], hidden_state])
        return model_out, [h]

    def get_initial_state(self):
        return [np.zeros(self.cell_size, np.float32)]


class TdtseRnnTorchModel(TorchModelV2, nn.Module):
    # TODO
    pass


class TdtseCnnTorchModel(TorchModelV2, nn.Module):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(TdtseCnnTorchModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        nn.Module.__init__(self)

        if not isinstance(obs_space, TDTSEStateSpace):
            obs_space = obs_space.original_space
            if not isinstance(obs_space, TDTSEStateSpace):
                raise Exception("obs nor obs.original_obs_space is of TDTSEStateSpace")

        config = TdtseCnnConfig(obs_space, model_config)
        self.config = config

        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.n_channel, config.filters_size, [8, 1]),
            torch.nn.ELU(),
            nn.Conv2d(config.filters_size, config.filters_size, [4, 1]),
            torch.nn.ELU(),
            nn.Conv2d(config.filters_size, config.filters_size, [2, 1]),
            torch.nn.ELU(),
            nn.Flatten()
        )
        
        conv_input_shape = (config.n_channel, config.n_sequence, config.n_lanes) # after transpose
        conv_output_shape = self.get_conv_output(conv_input_shape)
        l1_out_size = config.dense_layer_size_by_node * config.n_nodes

        from ray.rllib.models.torch.misc import normc_initializer, SlimFC
        
        self.l1 = SlimFC(conv_output_shape * config.n_nodes, l1_out_size, initializer=th_normc_initializer(1.0))
        self.logits = SlimFC(l1_out_size, num_outputs, initializer=th_normc_initializer(1.0))
        self.value_branch = SlimFC(l1_out_size, 1, initializer=th_normc_initializer(0.01))

    def forward(self, input_dict, state, seq_lens):
        input_data = input_dict["obs"] # shape: (B, num_lanes, num_history, num_channels), eg: (B, 4, 60, 3)
        input_data = input_data.transpose(1, 3) # channel first

        x = self.conv_layers(input_data)
        x = self.l1(x)
        model_out = self.logits(x)
        return model_out, state

    def get_conv_output(self, conv_input_shape):
        rand = torch.rand((1,) + conv_input_shape)
        out = self.conv_layers(rand) # flattened output
        return out.shape[1]

    def value_function(self):
        return self._value_out

    def metrics(self):
        return {"foo": tf.constant(42.0)}


class TdtseCnnTfModel(TFModelV2):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(TdtseCnnTfModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        if not isinstance(obs_space, TDTSEStateSpace):
            obs_space = obs_space.original_space
            if not isinstance(obs_space, TDTSEStateSpace):
                raise Exception("obs nor obs.original_obs_space is of TDTSEStateSpace")

        config = TdtseCnnConfig(obs_space, model_config)
        self.config = config
        self.add_noop_action = model_config['custom_model_config'].get('add_noop_action', False)

        self.tdtse = tf.keras.layers.Input(shape=obs_space["tdtse"].shape, name="tdtse")
        if config.use_progression:
            self.min_progression = tf.keras.layers.Input(shape=obs_space["min_progression"].shape,
                                                         name="min_progression")
            self.max_progression = tf.keras.layers.Input(shape=obs_space["max_progression"].shape,
                                                         name="max_progression")

        x = self.tdtse

        # TODO: with use_conv1d use conv1d layer with kernel=1.
        # TODO: hyperparameter run with much higher values of kernel size with maximum reaching 60.

        # num of filters increase with each conv2d layer.
        # reshape data for conv2d. eg: (1, 4, 60, 3) -> (4, 60, 3)
        reshape_layer = Reshape(target_shape=((config.n_lanes, config.n_sequence, config.n_channel)))
        conv_1_layer = tf.keras.layers.Conv2D(config.filters_size, [1, 8], activation=tf.nn.elu)
        conv_2_layer = tf.keras.layers.Conv2D(config.filters_size, [1, 4], activation=tf.nn.elu)
        conv_3_layer = tf.keras.layers.Conv2D(config.filters_size, [1, 2], activation=tf.nn.elu)
        flat_layer = tf.keras.layers.Flatten()

        nodes_features = []
        # extract tdtse for each agent
        splits = tf.split(self.tdtse, self.tdtse.shape[1], axis=1)
        for x in splits:  # apply convolution for each tdtse agent
            x = reshape_layer(x)
            x = conv_1_layer(x)
            x = conv_2_layer(x)
            x = conv_3_layer(x)
            x = flat_layer(x)
            nodes_features.append(x)

        # concatenate all convolutions output
        if config.n_nodes > 1:
            flat = tf.keras.layers.Concatenate(axis=-1)(nodes_features)
        else:
            flat = nodes_features[0]

        if config.use_progression:
            concat = tf.keras.layers.Concatenate(axis=-1)([flat, self.min_progression, self.max_progression])
        else:
            concat = flat

        layer_1 = tf.keras.layers.Dense(
            config.dense_layer_size_by_node * config.n_nodes,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(concat)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(1.0))(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)

        if config.use_progression:
            inputs = [self.tdtse, self.min_progression, self.max_progression]
        else:
            inputs = [self.tdtse]

        self.base_model = tf.keras.Model(
            inputs=inputs,
            outputs=[layer_out, value_out])

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        tdtse = input_dict["obs"]["tdtse"]

        if self.config.use_progression:
            min_progression = input_dict["obs"]["min_progression"]
            max_progression = input_dict["obs"]["max_progression"]
            inputs = [tdtse, min_progression, max_progression]
        else:
            inputs = [tdtse]
        model_out, self._value_out = self.base_model(inputs)

        if self.add_noop_action:
            action_mask = tf.cast(input_dict["obs"]["action_mask"], 'float32')
            inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
            model_out = model_out + inf_mask

        return model_out, state

    def value_function(self):
        return self._value_out

    def metrics(self):
        return {"foo": tf.constant(42.0)}


class TdtseCnnTfModelSoheil(TFModelV2):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(TdtseCnnTfModelSoheil, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        if not isinstance(obs_space, TDTSEStateSpace):
            obs_space = obs_space.original_space
            if not isinstance(obs_space, TDTSEStateSpace):
                raise Exception("obs nor obs.original_obs_space is of TDTSEStateSpace")

        config = TdtseCnnConfig(obs_space, model_config)
        self.config = config

        self.tdtse = tf.keras.layers.Input(shape=obs_space["tdtse"].shape, name="tdtse")
        if config.use_progression:
            self.min_progression = tf.keras.layers.Input(shape=obs_space["min_progression"].shape,
                                                         name="min_progression")
            self.max_progression = tf.keras.layers.Input(shape=obs_space["max_progression"].shape,
                                                         name="max_progression")

        x = self.tdtse

        # TODO: with use_conv1d use conv1d layer with kernel=1.
        # TODO: hyperparameter run with much higher values of kernel size with maximum reaching 60.

        # num of filters increase with each conv2d layer.
        # reshape data for conv2d. eg: (1, 4, 60, 3) -> (4, 60, 3)
        reshape_layer = Reshape(target_shape=((config.n_lanes, config.n_sequence, config.n_channel)))
        conv_1_layer = tf.keras.layers.Conv2D(config.filters_size, [2, 60], activation=tf.nn.elu)
        flat_layer = tf.keras.layers.Flatten()

        nodes_features = []
        # extract tdtse for each agent
        splits = tf.split(self.tdtse, self.tdtse.shape[1], axis=1)
        for x in splits:  # apply convolution for each tdtse agent
            x = reshape_layer(x)
            x = conv_1_layer(x)
            x = flat_layer(x)
            nodes_features.append(x)

        # concatenate all convolutions output
        if config.n_nodes > 1:
            flat = tf.keras.layers.Concatenate(axis=-1)(nodes_features)
        else:
            flat = nodes_features[0]

        if config.use_progression:
            concat = tf.keras.layers.Concatenate(axis=-1)([flat, self.min_progression, self.max_progression])
        else:
            concat = flat

        layer_1 = tf.keras.layers.Dense(
            config.dense_layer_size_by_node * config.n_nodes,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(concat)
        layer_2 = tf.keras.layers.Dense(
            config.dense_layer_size_by_node * config.n_nodes,
            name="my_layer2",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(layer_1)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(1.0))(layer_2)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_2)

        if config.use_progression:
            inputs = [self.tdtse, self.min_progression, self.max_progression]
        else:
            inputs = [self.tdtse]

        self.base_model = tf.keras.Model(
            inputs=inputs,
            outputs=[layer_out, value_out])

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        tdtse = input_dict["obs"]["tdtse"]

        if self.config.use_progression:
            min_progression = input_dict["obs"]["min_progression"]
            max_progression = input_dict["obs"]["max_progression"]
            inputs = [tdtse, min_progression, max_progression]
        else:
            inputs = [tdtse]
        model_out, self._value_out = self.base_model(inputs)
        return model_out, state

    def value_function(self):
        return self._value_out

    def metrics(self):
        return {"foo": tf.constant(42.0)}
