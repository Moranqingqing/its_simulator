import random

import tensorflow as tf
import numpy as np

from rltl.main.log import Log
# from rltl.utils.layers.spectral_normalisation import SpectralNormalization, SpectralNormalization2
# from rltl.utils.replay_memory import Memory
import tensorflow_addons as tfa

CLASSIC = "classic"
WASSERSTEIN = "wasserstein"
SUPER_GAN = "super_gan"
HYPER_GAN = "hyper_gan"
HYPER_NN = "hyper_nn"
AGGLOMERATED_GAN = "agglomerated_gan"
SPECTRAL_NORMALISATION = "spectral_normalisation"
print("tf.executing_eagerly()={}".format(tf.executing_eagerly()))
import logging

LOGGER = logging.getLogger(__name__)

log = Log(LOGGER)


# tf.keras.backend.set_floatx('float32')


def generator(gan_config, action_size, obs_size, plot_model_path=None):
    s = tf.keras.layers.Input(shape=[obs_size, ], name='s')
    a = tf.keras.layers.Input(shape=[action_size, ], name='a')
    inputs = [s, a]

    if gan_config["z_size"] > 0 and gan_config["type"] != HYPER_NN:
        z = tf.keras.layers.Input(shape=[gan_config["z_size"], ], name='z')
        inputs.append(z)

    if gan_config["type"] in [HYPER_GAN, HYPER_NN]:
        shape = [gan_config["e_size"], ]
        c = tf.keras.layers.Input(shape=shape, name='c')
        inputs.append(c)

    x = tf.keras.layers.concatenate(inputs)
    if "noisy_input_context_vector" in gan_config:
        noise = gan_config["noisy_input_context_vector"]
        if noise > 0:
            x = tf.keras.layers.GaussianNoise(noise)(x)
    for layer in gan_config["G_hidden_layers"]:
        x = tf.keras.layers.Dense(layer, activation=gan_config["G_activation"])(x)
        if gan_config["G_batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
        if "G_dropout" in gan_config:
            if gan_config["G_dropout"] > 0:
                x = tf.keras.layers.Dropout(gan_config["G_dropout"])(x)
        # x = tf.keras.layers.BatchNormalization()(x)

    if gan_config["model_key"] == "reward":
        outputs = tf.keras.layers.Dense(1)(x)
    elif gan_config["model_key"] == "dynamics":
        output_neurons = gan_config['G_output_neurons']
        output_activations = gan_config['G_output_activations']
        assert len(output_neurons) == len(output_activations)
        assert sum(output_neurons) == obs_size

        outputs = []
        for activation, n_neurons in zip(output_activations, output_neurons):
            output = tf.keras.layers.Dense(n_neurons, activation=activation)(x)
            outputs.append(output)
        if len(outputs) >= 2:
            outputs = tf.keras.layers.concatenate(outputs)
        elif len(outputs) == 1:
            outputs = outputs[0]  # otherwise keras will complain
        else:
            raise Exception()
    else:
        raise Exception()

    # if gan_config["z_size"] > 0:
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # else:
    #     model = tf.keras.Model(inputs=[s, a], outputs=last)
    if plot_model_path is not None:
        log.info("", "saving model architecture at {}".format(plot_model_path / "generator.png"))
        tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file=str(plot_model_path / "generator.png"))

    return model


def discriminator(gan_config, action_size, obs_size, plot_model_path=None):
    s = tf.keras.layers.Input(shape=[obs_size, ], name='s')
    a = tf.keras.layers.Input(shape=[action_size, ], name='a')
    s_ = tf.keras.layers.Input(shape=[obs_size, ], name='s_')
    r_ = tf.keras.layers.Input(shape=[1, ], name='r_')
    inputs = [s, a]

    if gan_config["model_key"] == "reward":
        inputs.append(r_)
    elif gan_config["model_key"] == "dynamics":
        inputs.append(s_)
    else:
        raise Exception()

    if gan_config["type"] in [HYPER_GAN, HYPER_NN]:
        shape = [gan_config["e_size"], ]
        c = tf.keras.layers.Input(shape=shape, name='c')
        inputs.append(c)

    x = tf.keras.layers.concatenate(inputs)
    use_spectral_normalization = WASSERSTEIN in gan_config and gan_config[WASSERSTEIN][
        "lipchitz_method"] == SPECTRAL_NORMALISATION
    for layer in gan_config["D_hidden_layers"]:
        l = tf.keras.layers.Dense(layer, activation=gan_config["D_activation"])
        if use_spectral_normalization:
            l = tfa.layers.SpectralNormalization(l)

        x = l(x)
        if gan_config["D_batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
        if "D_dropout" in gan_config:
            if gan_config["D_dropout"] > 0:
                x = tf.keras.layers.Dropout(gan_config["D_dropout"])(x)

    # x = tf.keras.layers.BatchNormalization()(x)
    gan_type = gan_config["type"]
    if gan_type == CLASSIC or gan_type in [HYPER_GAN, HYPER_NN]:
        if WASSERSTEIN in gan_config:
            l = tf.keras.layers.Dense(1)
            if use_spectral_normalization:
                l = tfa.layers.SpectralNormalization(l)
            last = l(x)
        else:
            last = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    else:
        raise Exception("Unknown GAN type= {}".format(gan_type))

    model = tf.keras.Model(inputs=inputs, outputs=last)
    if plot_model_path is not None:
        log.info("", "saving model architecture at {}".format(plot_model_path / "discriminator.png"))
        tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file=str(plot_model_path / "discriminator.png"))
    return model


def train_D(s, a, r_, s_, z, e, c, optimiser, G, D, gan_config, n_updates, epoch):
    a = tf.cast(a, tf.float32)
    s = tf.cast(s, tf.float32)
    s_ = tf.cast(s_, tf.float32)
    r_ = tf.cast(r_, tf.float32)
    if gan_config["z_size"] > 0 and gan_config["type"] != HYPER_NN:
        z = tf.cast(z, tf.float32)
    if gan_config["type"] == SUPER_GAN:
        e = tf.cast(e, tf.int64)

    if gan_config["type"] == HYPER_GAN:
        c = tf.cast(c, tf.float32)

    all_info = {
        "D_output_real_training": [],
        "D_output_fake_training": [],
    }

    for _ in range(n_updates):
        with tf.GradientTape() as disc_tape:
            g_inputs = [s, a]
            if gan_config["z_size"] > 0 and gan_config["type"] != HYPER_NN:
                g_inputs.append(z)

            if gan_config["type"] == HYPER_GAN:
                g_inputs.append(c)
            fake_out_ = G(g_inputs, training=True)  # for fake inference during training, dropout should be ON
            real_out = None
            if gan_config["model_key"] == "reward":
                real_out = r_
                real_out = tf.reshape(real_out, (-1, 1))

            elif gan_config["model_key"] == "dynamics":
                real_out = s_
            else:
                raise Exception()

            D_inputs_real = [s, a, real_out]
            D_inputs_fake = [s, a, fake_out_]

            if gan_config["type"] == HYPER_GAN:
                D_inputs_fake.append(c)
                D_inputs_real.append(c)

            real_D_ouput = D(D_inputs_real, training=True)
            fake_D_ouput = D(D_inputs_fake, training=True)
            loss, info = D_loss(
                e=e,
                real_D_ouput=real_D_ouput,
                fake_D_ouput=fake_D_ouput,
                gan_config=gan_config,
                epoch=epoch,
                D=D,
                real_out=real_out,
                fake_out=fake_out_,
                s=s, a=a, c=c)
        all_info["D_output_fake_training"].append(fake_D_ouput.numpy())
        all_info["D_output_real_training"].append(real_D_ouput.numpy())
        for k, v in info.items():
            if not k in all_info:
                all_info[k] = []
            all_info[k].append(v)
        discriminator_gradients = disc_tape.gradient(loss, D.trainable_variables)
        optimiser.apply_gradients(zip(discriminator_gradients, D.trainable_variables))
        if WASSERSTEIN in gan_config:

            lipchitz_method = gan_config[WASSERSTEIN]["lipchitz_method"]
            if lipchitz_method == "clipping":
                clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in D.weights]
                # pass
            elif lipchitz_method == "gradient_penalty":
                pass
            elif lipchitz_method == SPECTRAL_NORMALISATION:
                pass
            else:
                raise Exception("Unkwon lipchitz_method \"{}\"".format(lipchitz_method))

    return all_info


def train_G(s, a, r_, s_, z, e, c, lambda_, optimiser, G, D, n_updates, gan_config):
    a = tf.cast(a, tf.float32)
    s = tf.cast(s, tf.float32)
    s_ = tf.cast(s_, tf.float32)
    r_ = tf.cast(r_, tf.float32)
    if gan_config["z_size"] > 0 and gan_config["type"] != HYPER_NN:
        z = tf.cast(z, tf.float32)

    if gan_config["type"] in [HYPER_GAN, HYPER_NN]:
        c = tf.cast(c, tf.float32)
    for _ in range(n_updates):
        with tf.GradientTape() as gen_tape:
            inputs = [s, a]
            if gan_config["z_size"] > 0 and gan_config["type"] != HYPER_NN:
                inputs.append(z)

            if gan_config["type"] in [HYPER_GAN, HYPER_NN]:
                inputs.append(c)

            fake_out = G(inputs, training=True)
            if gan_config["type"] != HYPER_NN:
                fake_D_inputs = [s, a, fake_out]

                if gan_config["type"] == HYPER_GAN:
                    fake_D_inputs.append(c)
                fake_D_ouput = D(fake_D_inputs, training=True)
            else:
                fake_D_ouput = None
            real_out = None
            if gan_config["model_key"] == "reward":
                real_out = r_
            elif gan_config["model_key"] == "dynamics":
                real_out = s_
            else:
                raise Exception()

            loss, info = G_loss(fake_D_ouput=fake_D_ouput,
                                fake_out=fake_out,
                                real_out=real_out,
                                e=e,
                                lambda_=lambda_,
                                gan_config=gan_config)

        generator_gradients = gen_tape.gradient(loss, G.trainable_variables)
        optimiser.apply_gradients(zip(generator_gradients, G.trainable_variables))
    return info


binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.CategoricalCrossentropy()


def gradient_penalty(D, batch_size, real_images, fake_images, s, a, c, gan_config):
    """ Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    alpha = tf.random.uniform([], 0, 1)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    inputs = [s, a, interpolated]

    if gan_config["type"] == HYPER_GAN:
        inputs.append(c)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = D(inputs, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
    gp = tf.reduce_mean((norm - 1.0) ** 2)

    # epsilon = tf.random_uniform([], 0.0, 1.0)
    # x_hat = real_images * epsilon + (1 - epsilon) * gen_images
    # d_hat = netD(x_hat, y, BATCH_SIZE, LOSS, reuse=True)
    # gradients = tf.gradients(d_hat, x_hat)[0]
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    # gradient_penalty = 10 * tf.reduce_mean((slopes - 1.0) ** 2)

    return gp


def D_loss(e, real_D_ouput, fake_D_ouput, gan_config, epoch, D=None, real_out=None, fake_out=None, s=None, a=None,
           c=None):
    """
    for notation, cf paper https://arxiv.org/pdf/1704.00028.pdf
    """
    info = {}
    type = gan_config["type"]
    if type == CLASSIC or type == HYPER_GAN:
        if WASSERSTEIN in gan_config:
            critic_real = tf.reduce_mean(real_D_ouput)
            critic_fake = tf.reduce_mean(fake_D_ouput)
            info["critic_real"] = critic_real.numpy()
            info["critic_fake"] = critic_fake.numpy()
            lipchitz_method = gan_config[WASSERSTEIN]["lipchitz_method"]
            if lipchitz_method == "gradient_penalty":
                gp = gradient_penalty(D, len(real_out), real_out, fake_out, s, a, c, gan_config)
                lambda_gp = gan_config[WASSERSTEIN]["gp_lambda"]
                info["gradient_penalty"] = gp.numpy()
            else:
                gp = 0
                lambda_gp = 0
            loss = -(critic_real - critic_fake) + lambda_gp * gp
        else:
            real_loss = tf.reduce_mean(tf.math.log(real_D_ouput))
            fake_loss = tf.reduce_mean(tf.math.log(1. - fake_D_ouput))
            info["E(log(D(x_real))"] = real_loss.numpy()
            info["E(log(1 - D(x_fake))"] = fake_loss.numpy()
            loss = -(real_loss + fake_loss)
    else:
        raise Exception("Unkown type \"{}\"".format(type))

    info["D_loss"] = loss.numpy()

    return loss, info


def G_loss(fake_D_ouput, fake_out, real_out, e, gan_config, lambda_=0):
    type = gan_config["type"]
    info = {}
    if type in [CLASSIC, HYPER_GAN]:
        if WASSERSTEIN in gan_config:
            ED_x_fake = tf.reduce_mean(fake_D_ouput)
            # generator try to maximise the D output for fake instances
            loss = - ED_x_fake
        else:
            if "non_saturing" in gan_config and gan_config["non_saturing"]:
                # Non-Saturating GAN Loss:
                #   generator: maximize log(D(G(z)))
                ElogD_x_fake = tf.reduce_mean(tf.math.log(fake_D_ouput))
                loss = - ElogD_x_fake
            else:
                # saturated loss
                ElogD_x_fake = tf.reduce_mean(tf.math.log(1. - fake_D_ouput))
                loss = ElogD_x_fake


    elif type == HYPER_NN:
        loss = tf.reduce_mean(tf.abs(real_out - fake_out))
    else:
        raise Exception("Unkown type \"{}\"".format(type))
    info["G_loss"] = loss.numpy()
    return loss, info


def train(real_out_str, train_ds, test_ds, lambda_, G, D, G_opti, D_opti, gan_config, D_n_updates, G_n_updates,
          log_folder, obs_size, stop, interval_eval_epoch, callback=None, verbose=True):
    # train_ds = train_ds.batch(gan_config["batch_size"])
    if gan_config["batch_size"] == "max":
        batch_size = len(train_ds)
    else:
        batch_size = gan_config["batch_size"]

    minibatches = train_ds.batch(batch_size)

    all_train_samples = train_ds.batch(len(list(train_ds)))
    all_tests_samples = test_ds.batch(len(list(test_ds)))

    plots, writers = create_plots_and_writers(gan_config, real_out_str, log_folder)
    done = False
    epoch = 0
    while not done:
        done = perform_an_epoch(plots=plots,
                                minibatches=minibatches,
                                writers=writers,
                                real_out_str=real_out_str,
                                epoch=epoch, lambda_=lambda_,
                                G=G, D=D, G_opti=G_opti, D_opti=D_opti,
                                gan_config=gan_config, D_n_updates=D_n_updates,
                                G_n_updates=G_n_updates,
                                obs_size=obs_size, stop=stop,
                                all_train_samples=all_train_samples,
                                all_tests_samples=all_tests_samples,
                                callback=callback,
                                verbose=verbose,
                                interval_eval_epoch=interval_eval_epoch)
        epoch += 1


def create_plots_and_writers(gan_config, real_out_str, log_folder):
    plots = {}
    plots["testing_loss"] = "training & testing"
    plots["G_loss"] = "G & D losses"
    plots["D_loss"] = "G & D losses"
    plots["training_loss"] = "training & testing"
    plots["D_output_real_training"] = "D output"
    plots["D_output_fake_training"] = "D output"
    plots["D_output_real_testing"] = "D output"
    plots["D_output_fake_testing"] = "D output"
    if WASSERSTEIN in gan_config:
        key = "minmax wgan"  # E(D(x_real)) - E(D(x_fake))"
        plots["critic_real"] = key
        plots["critic_fake"] = key
        if gan_config[WASSERSTEIN]["lipchitz_method"] == "gradient_penalty":
            plots["gradient_penalty"] = "gradient_penalty"
    else:
        key = "minmax gan"  # E(log(D(x_real)) + E(log(1 - D(x_fake))"
        plots["E(log(D(x_fake))"] = key
        plots["E(log(D(x_real))"] = key
        plots["E(log(1 - D(x_fake))"] = key

    writers = {}
    for metric, _ in plots.items():
        writers[metric] = tf.summary.create_file_writer(str(log_folder / metric))

    return plots, writers


def perform_an_epoch(plots, minibatches, writers, real_out_str,
                     epoch, all_train_samples, all_tests_samples, lambda_,
                     G, D, G_opti, D_opti,
                     gan_config, D_n_updates,
                     G_n_updates,
                     obs_size, stop, callback=None, verbose=True, interval_eval_epoch=100):
    done = False
    isnan = False
    if "max_epochs" in stop:
        if epoch > stop["max_epochs"]:
            done = True

    lentrain = None
    # for n, (s, a, r_, s_, e, c) in all_train_samples.enumerate():
    for n_minibatch, (s, a, r_, s_, e, c) in minibatches.enumerate():
        if gan_config["z_size"] > 0 and gan_config["type"] != HYPER_NN:
            z = np.random.normal(-1.0, 1.0, size=[len(s), gan_config["z_size"]])
        else:
            z = None
        lentrain = len(s)
        if gan_config["type"] != HYPER_NN:
            info_D = train_D(s, a, r_, s_, z,
                             e=e,
                             c=c,
                             optimiser=D_opti,
                             G=G, D=D,
                             gan_config=gan_config, epoch=epoch,
                             n_updates=D_n_updates)
        info_G = train_G(s, a, r_, s_, z,
                         e=e,
                         c=c,
                         optimiser=G_opti,
                         lambda_=lambda_,
                         G=G, D=D,
                         gan_config=gan_config,
                         n_updates=G_n_updates)
        if n_minibatch % gan_config["interval_verbose_minibatch"] == 0:
            d_loss = -1
            if gan_config["type"] != HYPER_NN:
                d_loss = info_D["D_loss"]

            print("[EPOCH={}][MINIBATCH={}][STEP={}] loss_G={} loss_D={}".format(
                epoch, n_minibatch,
                n_minibatch + len(list(minibatches)) * epoch,
                info_G["G_loss"], d_loss
            ))
            # if verbose:
            for k, v in info_G.items():
                with writers[k].as_default():
                    tf.summary.scalar(str(plots[k]), data=v, step=epoch)
            writers[k].flush()

            if gan_config["type"] != HYPER_NN:
                for k, v in info_D.items():
                    with writers[k].as_default():
                        tf.summary.scalar(str(plots[k]), data=np.mean(v), step=epoch)

                writers[k].flush()

    training_loss, testing_loss = None, None
    if interval_eval_epoch is not None and epoch % interval_eval_epoch == 0:
        log.info("", "[{}] EPOCH={}".format(gan_config["type"], epoch))
        testing = []
        training = []
        testing_not_reduced = []
        acc_on_random_s_ = []
        acc_on_real_s_ = []
        acc_on_fake_s_ = []
        for ds, losses in [(all_train_samples, training), (all_tests_samples, testing)]:
            if ds is not None:
                for (s, a, r_, s_, e, c) in ds:
                    a = tf.cast(a, tf.float32)
                    s = tf.cast(s, tf.float32)
                    s_ = tf.cast(s_, tf.float32)
                    r_ = tf.cast(r_, tf.float32)
                    e = tf.cast(e, tf.int64)
                    c = tf.cast(c, tf.float32)
                    inputs = [s, a]
                    if gan_config["z_size"] > 0 and gan_config["type"] != HYPER_NN:
                        z = np.random.normal(-1.0, 1.0, size=[len(s), gan_config["z_size"]])
                        inputs.append(z)
                    if gan_config["type"] in [HYPER_GAN, HYPER_NN]:
                        inputs.append(c)

                    fake_out = G(inputs, training="training" in gan_config and gan_config["training"])

                    if gan_config["model_key"] == "reward":
                        real_out = r_
                        real_out = tf.reshape(real_out, (-1, 1))
                        real_out_str = "r_"
                        out_size = 1
                    elif gan_config["model_key"] == "dynamics":
                        real_out = s_
                        real_out_str = "s_"
                        out_size = obs_size
                    else:
                        raise Exception()

                    t_loss = tf.reduce_mean(tf.abs(real_out - fake_out))
                    losses.append(t_loss)
                    if np.isnan(t_loss):
                        log.warn("", "is NAN !!!!")
                        isnan = True

        testing_loss = np.mean(testing)
        training_loss = np.mean(training)

        testing_not_reduced.append(tf.abs(real_out - fake_out))
        input_D_random = [s, a, 100 * np.random.random(size=[len(real_out), out_size])]
        input_D_real = [s, a, real_out]
        input_D_fake = [s, a, fake_out]
        if gan_config["type"] in [HYPER_GAN, HYPER_NN]:
            input_D_random.append(c)
            input_D_fake.append(c)
            input_D_real.append(c)
        if gan_config["type"] != HYPER_NN:
            # d_out_random = D(input_D_random, training="training" in gan_config and gan_config["training"])
            d_real = D(input_D_real, training="training" in gan_config and gan_config["training"])
            d_fake = D(input_D_fake, training="training" in gan_config and gan_config["training"])
            if "factorize_fake_output" in gan_config and gan_config["factorize_fake_output"] and gan_config[
                "type"] == SUPER_GAN:
                # _, t_random = d_out_random
                _, t_real = d_real
                _, t_fake = d_fake
            else:
                # t_random = d_out_random
                t_real = d_real
                t_fake = d_fake
            # acc_on_random_s_.append(np.mean(t_random))
            acc_on_real_s_.append(np.mean(t_real))

            l1_by_env = {}

            for i in range(min(200, len(a))):
                if not (e[i].numpy() in l1_by_env):
                    l1_by_env[e[i].numpy()] = []

                l1_by_env[e[i].numpy()].append(np.abs(fake_out[i].numpy() - real_out[i].numpy()))
            log.info("", "---------- l1 by env -----------")
            for ei in range(len(l1_by_env)):
                if ei in l1_by_env:
                    log.info("", "env={} -> test set l1={}".format(ei, np.mean(l1_by_env[ei], axis=0)))
            if gan_config["type"] != HYPER_NN:
                acc_on_fake_s_.append(np.mean(t_fake))
        info = {
            "testing_loss": testing_loss,
            "training_loss": training_loss,

        }
        if gan_config["type"] != HYPER_NN:
            info["D_output_real_testing".format(real_out_str)] = np.mean(acc_on_real_s_)
            info["D_output_fake_testing".format(real_out_str)] = np.mean(acc_on_fake_s_)

        for k, v in info.items():
            with writers[k].as_default():
                tf.summary.scalar(str(plots[k]), data=v, step=epoch)

        if "testing_loss_treshold" in stop:
            if info["testing_loss"] < stop["testing_loss_treshold"]:
                done = True
        if "training_loss_treshold" in stop:
            if info["training_loss"] < stop["training_loss_treshold"]:
                done = True
        done = done or isnan

    if callback:
        callback(epoch, verbose, training_loss, testing_loss)

    return done
