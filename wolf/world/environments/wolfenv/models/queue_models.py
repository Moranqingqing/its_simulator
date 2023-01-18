from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf, try_import_torch


tf = try_import_tf()
torch = try_import_torch()


class QueueObsModel(TFModelV2):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(QueueObsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        raise Exception("Deprecated, must modify for multinodes and mono node usage")
        # Define the core model layers which will be used by the other
        # output heads of DistributionalQModel
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        conv_1 = tf.keras.layers.Conv2D(32, [8, 1], activation=tf.nn.elu)(self.inputs)
        conv_2 = tf.keras.layers.Conv2D(32, [4, 1], activation=tf.nn.elu)(conv_1)
        conv_3 = tf.keras.layers.Conv2D(32, [2, 1], activation=tf.nn.elu)(conv_2)
        flat = tf.keras.layers.Flatten()(conv_3)

        layer_1 = tf.keras.layers.Dense(128, name="my_layer1", activation=tf.nn.relu,
                                        kernel_initializer=normc_initializer(1.0))(flat)
        layer_out = tf.keras.layers.Dense(num_outputs, name="my_out", activation=None,
                                          kernel_initializer=normc_initializer(1.0))(layer_1)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None,
                                          kernel_initializer=normc_initializer(0.01))(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return self._value_out

    def metrics(self):
        return {"foo": tf.constant(42.0)}
