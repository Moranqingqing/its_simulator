from abc import abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import logging

from rltl.main.log import Log

LOGGER = logging.getLogger(__name__)

log = Log(LOGGER)

SPIDER = "spider"


class Spider(tf.keras.Model):
    """Conditional variational autoencoder."""

    @abstractmethod
    def create_encoder(self, inputs):
        pass

    @abstractmethod
    def create_decoder(self, inputs):
        pass

    def __init__(self, action_size, obs_size, source_tasks_number, config, plot_model_path=None):
        super(Spider, self).__init__()
        self.latent_dim = config["z_size"]
        self.action_size = action_size
        self.obs_size = obs_size
        self.config = config
        self.source_tasks_number = source_tasks_number

        s = tf.keras.layers.Input(shape=[self.obs_size, ], name='s')
        a = tf.keras.layers.Input(shape=[self.action_size, ], name='a')
        self.encoders = {}
        for task in range(self.source_tasks_number):
            task_encoder = tf.keras.Model(name="Encoder({})".format(task),
                                          inputs=[s, a],
                                          outputs=self.create_encoder([s, a]))
            self.encoders[task] = task_encoder

        if plot_model_path is not None:
            for task in range(self.source_tasks_number):
                print("",
                         "saving model architecture at {}".format(plot_model_path / "encoder_task={}.png".format(task)))
                tf.keras.utils.plot_model(self.encoders[task], show_shapes=True, dpi=64,
                                          to_file=str(plot_model_path / "encoder_task={}.png".format(task)))

        z = tf.keras.layers.Input(shape=[self.latent_dim, ], name='z')
        self.decoder = tf.keras.Model(inputs=[z], outputs=self.create_decoder([z]))

        if plot_model_path is not None:
            print("", "saving model architecture at {}".format(plot_model_path / "decoder.png"))
            tf.keras.utils.plot_model(self.decoder, show_shapes=True, dpi=64,
                                      to_file=str(plot_model_path / "decoder.png"))

    @tf.function
    def sample(self, s, a, id_task):
        z = self.encode(s, a, id_task)
        x = self.decode(z)
        return x

    def encode(self, s, a, id_task):
        inputs = [s, a]
        mean, logvar = tf.split(self.encoders[id_task](inputs), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):  # , apply_sigmoid=False):
        fake_s_ = self.decoder([z])
        return fake_s_

    def gaussian_nll_tf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def gaussian_nll(self, mu, log_sigma, x):
        return 0.5 * ((x - mu) / tf.exp(log_sigma)) ** 2 + log_sigma + 0.5 * np.log(2 * np.pi)

    def compute_loss(self, s, a, s_, cvae_config, c=None, batch_size=None):
        z_mu, z_log_sigma_sq = self.encode(s, a, s_, c)
        z = self.reparameterize(z_mu, z_log_sigma_sq)
        s_pred = self.decode(s, a, z, c)
        type_rec_loss = cvae_config["reconstruction_loss"]["type"]
        params_rec_loss = cvae_config["reconstruction_loss"]["params"]
        info = {}

        if type_rec_loss == "gaussian_log_likelihood" and params_rec_loss["std"] == "optimal_sigma":
            log_sigma = tf.math.log(tf.sqrt(tf.reduce_mean((s_ - s_pred) ** 2, [0, 1], keepdims=True)))
            # info["optimal_sigma"] = log_sigma
            # log_sigma = tf.math.log(tf.sqrt(tf.reduce_mean((s_ - s_pred) ** 2, [0, 1, 2, 3], keepdims=True)))
            reconstruction_loss = tf.reduce_sum(self.gaussian_nll(s_pred, log_sigma, s_))
            kl_loss = -tf.reduce_sum(0.5 * (1 + z_log_sigma_sq - z_mu ** 2 - tf.exp(z_log_sigma_sq)))
            loss = (reconstruction_loss + kl_loss) / np.sum(s_.shape)
        else:

            # KL loss
            logpz = self.gaussian_nll_tf(z, 0., 0.)
            logqz_x = self.gaussian_nll_tf(z, z_mu, z_log_sigma_sq)
            kl_loss = tf.reduce_mean(logpz - logqz_x)

            # seen output as proba or direct reconstruction:
            # https://stats.stackexchange.com/questions/368001/is-the-output-of-a-variational-autoencoder-meant-to-be-a
            # -distribution-that-can-b
            if type_rec_loss == "mse":
                # just the mse instead of gaussian log likehood, kinda equivalent
                # https://towardsdatascience.com/variational-autoencoders-vaes-for-dummies-step-by-step-tutorial-69e6d1c9d8e9
                reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(s_, s_pred))
            elif type_rec_loss == "cross_entropy":
                # output probability, and interprete the proba as a state (work with black and white images basically,
                # cf all VAE+MNIST implementations)
                reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=s_, y_pred=s_pred))
            elif type_rec_loss == "gaussian_log_likelihood":
                if "decoder_output_std" in params_rec_loss and params_rec_loss["decoder_output_std"]:
                    # original implementation where mu and std are learned, it has been shown it doesn't work
                    #  https://discuss.pytorch.org/t/multivariate-gaussian-variational-autoencoder-the-decoder-part/58235/5
                    mean_s_pred, logvar_s_pred = tf.split(s_pred, num_or_size_splits=2, axis=1)
                    reconstruction_loss = \
                        tf.math.log(2 * np.pi) + logvar_s_pred + (mean_s_pred - s_) ** 2 / (
                                2 * tf.math.exp(logvar_s_pred))
                    reconstruction_loss = tf.reduce_mean(reconstruction_loss)
                else:
                    # std is fixed, decoder outputs mu only
                    # https://github.com/wuga214/IMPLEMENTATION_Variational-Auto-Encoder/blob/master/models/vae.py
                    std = params_rec_loss["std"]
                    reconstruction_loss = tf.reduce_mean(
                        0.5 * tf.square(s_ - s_pred) / (2 * tf.square(std)) + tf.math.log(std))
            else:
                raise Exception("Unknown loss type: {}".format(cvae_config["reconstruction_loss"]))

            if "kl_weight" not in cvae_config or cvae_config["kl_weight"] == "auto":
                kl_weight = 1. / batch_size
            else:
                kl_weight = cvae_config["kl_weight"]

            if "reconstruction_weight" not in cvae_config or cvae_config["reconstruction_weight"] == "auto":
                # https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
                # https://keras.io/examples/generative/vae/ reconstruction_loss
                # https://pyod.readthedocs.io/en/latest/_modules/pyod/models/vae.html
                reconstruction_weight = self.obs_size / 2  # /2
            else:
                reconstruction_weight = float(cvae_config["reconstruction_weight"])

            # free_bit trick, Enforcing Latent Variable Use
            if "lambda_kl_free_bit" not in cvae_config:
                lambda_kl_free_bit = -np.inf
            else:
                lambda_kl_free_bit = float(cvae_config["lambda_kl_free_bit"])

            # TODO instead of free_bit, use scheduler
            # TODO normalising flow

            loss = reconstruction_weight * reconstruction_loss - kl_weight * tf.math.maximum(lambda_kl_free_bit,
                                                                                             kl_loss)
        info["reconstruction_loss"] = reconstruction_loss  # .numpy()
        info["kl_loss"] = kl_loss  # .numpy()
        info["vae"] = loss  # .numpy()
        return loss, info

    @tf.function
    def train_step(self, s, a, r_, s_, e, c, optimizer, cvae_config, batch_size):
        a = tf.cast(a, tf.float32)
        s = tf.cast(s, tf.float32)
        s_ = tf.cast(s_, tf.float32)
        r_ = tf.cast(r_, tf.float32)
        if cvae_config["type"] == HYPER_VAE:
            c = tf.cast(c, tf.float32)
        else:
            c = None
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss, info = self.compute_loss(s, a, s_, cvae_config=cvae_config, c=c, batch_size=batch_size)
        # print(model.trainable_variables)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, info

    def create_plots_and_writers(self, gan_config, real_out_str, log_folder):
        plots = {}
        plots["testing_loss"] = "training & testing"
        plots["loss_minibatches"] = "training & testing"
        plots["training_loss"] = "training & testing"
        plots["vae"] = "losses"
        plots["kl_loss"] = "losses"
        plots["reconstruction_loss"] = "losses"
        writers = {}
        for metric, _ in plots.items():
            writers[metric] = tf.summary.create_file_writer(str(log_folder / metric))

        return plots, writers

    def train(self,
              optimizer, train_ds, test_ds,
              stop, cvae_config, real_out_str, log_folder,
              callback, verbose, interval_eval_epoch, obs_size):
        if cvae_config["batch_size"] == "max":
            batch_size = len(train_ds)
        else:
            batch_size = cvae_config["batch_size"]

        minibatches = train_ds.batch(batch_size)
        print("Number of minibaches (ie number of gradient step per epoch): {}".format(len(minibatches)))
        all_train_samples = train_ds.batch(len(list(train_ds)))
        all_tests_samples = test_ds.batch(len(list(test_ds)))

        plots, writers = self.create_plots_and_writers(cvae_config, real_out_str, log_folder)
        done = False
        epoch = 0
        while not done:
            done = self.perform_an_epoch(
                optimizer=optimizer,
                plots=plots,
                minibatches=minibatches,
                writers=writers,
                real_out_str=real_out_str,
                epoch=epoch,
                cvae_config=cvae_config,
                obs_size=obs_size, stop=stop,
                all_train_samples=all_train_samples,
                all_tests_samples=all_tests_samples,
                callback=callback,
                verbose=verbose,
                interval_eval_epoch=interval_eval_epoch)
            epoch += 1
            if "max_epochs" in stop:
                if epoch > stop["max_epochs"]:
                    done = True

    def perform_an_epoch(self,
                         optimizer, plots, minibatches, writers, real_out_str,
                         epoch, all_train_samples, all_tests_samples,
                         cvae_config,
                         obs_size, stop, callback=None, verbose=True, interval_eval_epoch=100):
        # start_time = time.time()
        losses = {
            "vae": 0,
            "kl_loss": 0,
            "reconstruction_loss": 0
        }
        batch_size = len(all_train_samples)
        for n_minibatch, (s, a, r_, s_, e, c) in minibatches.enumerate():
            # for train_x, train_y in train_dataset:
            if cvae_config["type"] != HYPER_VAE:
                c = None
            loss, info = self.train_step(s, a, r_, s_, e, c, optimizer, cvae_config, batch_size)
            for k, v in info.items():
                losses[k] += v
                with writers[k].as_default():
                    tf.summary.scalar(str(plots[k]), data=v, step=epoch)

        print("[epoch={},lr={:.4f}]{}".format(
            epoch,
            optimizer._decayed_lr(tf.float32).numpy(),
            " ".join(["{}={:.6f}".format(k, v / len(minibatches)) for k, v in losses.items()])))

        losses = {
            "training_loss": None,
            "testing_loss": None
        }

        if epoch % interval_eval_epoch == 0:

            for name, samples in [("training_loss", all_train_samples), ("testing_loss", all_tests_samples)]:
                print("---------- {} samples --------------".format(name))
                print("----------------------------------------")

                for i, (s, a, r_, s_, e, c) in enumerate(samples):
                    if cvae_config["type"] != HYPER_VAE:
                        c = None
                    all_fake_s_ = self.sample(s, a, c).numpy()
                    # z = tf.cast(tf.random.normal(shape=(len(s), cvae_config["z_size"])), tf.float32)
                    # all_fake_decoder_ = model.decoder([s, a, z]).numpy()
                    all_s_ = s_.numpy()
                    if i > 0:
                        raise Exception("")
                    a = tf.cast(a, tf.float32)
                    s = tf.cast(s, tf.float32)
                    s_ = tf.cast(s_, tf.float32)
                    r_ = tf.cast(r_, tf.float32)
                    if c is not None:
                        c = tf.cast(c, tf.float32)
                    losses[name], _ = self.compute_loss(s, a, s_, cvae_config=cvae_config, c=c,
                                                        batch_size=len(all_train_samples))

                for i, (fake_s_, s_) in enumerate(zip(all_fake_s_, all_s_)):
                    print(s_, fake_s_)
                    if i > 20:
                        break
                print("----------------------------------------")
                with writers[name].as_default():
                    tf.summary.scalar(str(plots[name]), data=np.mean(losses[name]), step=epoch)
            print("------------------")
            for k, v in losses.items():
                print("{}={}".format(k, v))
            print("------------------")
            if "show_latent_space" in cvae_config and cvae_config["show_latent_space"]:
                for i, (s, a, r_, s_, e, c) in enumerate(all_train_samples):
                    mu_z, sigma_z = self.encode(s, a, c, s_)
                    gaussian = tfp.distributions.Normal(loc=mu_z, scale=sigma_z)
                    z = tf.squeeze(gaussian.sample(1))
                    import matplotlib.pyplot as plt
                    edge_colors = ["blue", "green", "red", "yellow"]
                    for k in range(len(z)):
                        # plt.scatter(z[:,0],z[:,1],alpha=0.1)
                        sk = s[k]
                        ck = c[k]
                        ck = np.where(ck == 1)
                        zk = z[k]
                        if sk[0] < 0.5 and sk[1] < 0.5:
                            color = "blue"
                        elif sk[0] >= 0.5 and sk[1] < 0.5:
                            color = "red"
                        elif sk[0] >= 0.5 and sk[1] >= 0.5:
                            color = "green"
                        else:
                            color = "yellow"
                        plt.scatter(zk[0], zk[1], alpha=0.25, color=color, edgecolors=edge_colors[ck[0][0]])
                    plt.show()
        callback(epoch, True, losses["training_loss"], losses["testing_loss"])

        return False


# class SamplerSpider(Spider):
#     def __init__(self, decoder, config, plot_model_path=None):
#         super(SamplerCVAE, self).__init__(action_size, obs_size, source_tasks_number, config, plot_model_path=None)


class DenseSpider(Spider):

    def __init__(self,
                 action_size, obs_size,
                 encoder_hiddens,
                 decoder_hiddens,
                 encoder_activation, decoder_activation, config, source_tasks_number, plot_model_path=None):
        self.encoder_hiddens = encoder_hiddens
        self.decoder_hiddens = decoder_hiddens
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        super(DenseSpider, self).__init__(action_size=action_size, obs_size=obs_size,
                                          config=config,
                                          source_tasks_number=source_tasks_number,
                                          plot_model_path=plot_model_path)

    def create_encoder(self, inputs):
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = tf.keras.layers.concatenate(inputs)
        for layer in self.encoder_hiddens:
            x = tf.keras.layers.Dense(layer, activation=self.encoder_activation)(x)
        encoder = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(x)
        return encoder

    def create_decoder(self, inputs):
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = tf.keras.layers.concatenate(inputs)

        for layer in self.decoder_hiddens:
            x = tf.keras.layers.Dense(layer, activation=self.encoder_activation)(x)
        params_rec = self.config["reconstruction_loss"]["params"]
        type_rec = self.config["reconstruction_loss"]["type"]
        if type_rec == "gaussian_log_likelihood":
            if "decoder_output_std" in params_rec and params_rec["decoder_output_std"]:
                decoder = tf.keras.layers.Dense(self.obs_size * 2)(x)
                mean_s_pred, logvar_s_pred = tf.split(decoder, num_or_size_splits=2, axis=1)
                if "sigmoid_output" in self.config and self.config["sigmoid_output"]:
                    mean_s_pred = tf.sigmoid(mean_s_pred)
                decoder = tf.keras.layers.concatenate([mean_s_pred, logvar_s_pred])
            else:
                decoder = tf.keras.layers.Dense(self.obs_size)(x)
                if "sigmoid_output" in self.config and self.config["sigmoid_output"]:
                    decoder = tf.sigmoid(decoder)

        else:
            decoder = tf.keras.layers.Dense(self.obs_size)(x)
            if "sigmoid_output" in self.config and self.config["sigmoid_output"]:
                decoder = tf.sigmoid(decoder)

        return decoder
