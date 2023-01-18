import tensorflow as tf
import numpy as np

from tensorflow.keras import Model

from rltl.main.policy import Policy, EpsilonGreedyPolicy


class FeedForwardQModel(Model):
    def __init__(self, layers=None):
        super(FeedForwardQModel, self).__init__()
        if layers is None:
            layers = [100]
        self.layers_ = []
        for l in layers:
            self.layers_.append(tf.keras.layers.Dense(l, activation='relu'))
        self.value = tf.keras.layers.Dense(2)

        # TODO
        # replace 2 by number of action
        # add softmax to each head
        # use 0.9 gamma
        # custom activation (leakyrelu ...) etc

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.layers_:
            x = layer(x)
        x = self.value(x)
        return x


class DQNPolicy(Policy):

    def __init__(self, action_space, dqn_model):
        Policy.__init__(self, action_space)
        self.dqn_model = dqn_model

    def act(self, s, batch_mode):
        s = tf.cast(s, dtype=tf.float32)
        if batch_mode:
            q_value = self.dqn_model(tf.convert_to_tensor(s, dtype=tf.float32))
            action = tf.math.argmax(q_value, axis=1)
            # print("hello")
        else:
            q_value = self.dqn_model(tf.convert_to_tensor([s], dtype=tf.float32))[0]
            action = np.argmax(q_value)
        return action


class DQNAgent:
    def __init__(self, action_space,
                 optimizer=None,
                 gamma=0.999,
                 update_network_frequency=10, model_creator=None):  # , memory):
        if model_creator == None:
            model_creator = lambda: FeedForwardQModel()

        if optimizer is None:
            optimizer = {"name": "adam", "params": {"lr": 1e-3}}
        self.gamma = gamma

        self.Q = model_creator()
        self.greedy_policy = DQNPolicy(action_space, self.Q)
        self.egreedy_policy = EpsilonGreedyPolicy(action_space, self.greedy_policy)
        self.Q_target = model_creator()
        # self.opt = optimizers.Adam(lr=self.lr, )
        if optimizer["name"] == "adam":
            # self.opt = tf.keras.optimizers.Adam(**optimizer["params"])
            self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=optimizer["params"]["lr"])
        elif optimizer["name"] == "rms":
            self.opt = tf.keras.optimizers.RMSprop(**optimizer["params"])
        elif optimizer["name"] == "sgd":
            self.opt = tf.keras.optimizers.SGD(**optimizer["params"])
        else:
            raise Exception()
        self.td_errors_loss_fn = lambda x, y: \
            tf.compat.v1.losses.mean_squared_error(
                x, y,
                reduction=tf.compat.v1.losses.Reduction.NONE)

        # self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        # self.state_size = 4
        self.action_size = 2
        self.train_iteration = 0
        self.update_network_frequency = update_network_frequency

    def update_target(self):
        self.Q_target.set_weights(self.Q.get_weights())

    def train(self, mini_batch):
        error = self.update(mini_batch)
        if self.train_iteration % self.update_network_frequency == 0:
            self.update_target()
        return error

    def update(self, mini_batch):

        states = [i[0] for i in mini_batch]
        actions = [i[1] for i in mini_batch]
        rewards = [i[2] for i in mini_batch]
        next_states = [i[3] for i in mini_batch]
        dones = [i[4] for i in mini_batch]

        dqn_variable = self.Q.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)

            s = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
            s_ = tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)
            r_ = tf.convert_to_tensor(rewards, dtype=tf.float32)
            a = tf.convert_to_tensor(actions, dtype=tf.int32)
            valid_mask = (1 - tf.convert_to_tensor(dones, dtype=tf.float32))

            target_q = self.Q_target(s_)
            next_action = tf.argmax(target_q, axis=1)
            max_q_ = tf.reduce_sum(tf.one_hot(next_action, self.action_size, dtype=tf.float32) * target_q, axis=1)

            td_target = valid_mask * self.gamma * max_q_ + r_
            # print(td_target)
            q_values = tf.reduce_sum(tf.one_hot(a, self.action_size, dtype=tf.float32) * self.Q(s), axis=1)

            # td_loss = tf.square(main_value - target_value)
            # td_loss = tf.reduce_mean(td_loss)
            # td_loss = (1 - dones) * self.td_errors_loss_fn(q_values, td_target)
            td_loss = self.td_errors_loss_fn(q_values, td_target)
            td_loss = tf.reduce_sum(input_tensor=td_loss)

        dqn_grads = tape.gradient(td_loss, dqn_variable)
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

        return td_loss
