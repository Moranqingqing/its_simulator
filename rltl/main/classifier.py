import tensorflow as tf
import numpy as np
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import ExponentialDecay
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from rltl.utils.bayesian.dense_variational import DenseVariational

print("tf.executing_eagerly()={}".format(tf.executing_eagerly()))

CLASSIFIER = "classifier"
BAYESIAN_CLASSIFIER = "bayesian_classifier"


def show_schedule(type, params):
    schedule = create_schedule(type, params)
    y = []
    n = 100000
    for step in range(n):
        y.append(schedule(step))
    import matplotlib.pyplot as plt
    plt.plot(range(n), y)
    plt.show()


def create_schedule(type, params):
    if type == "ExponentialDecay":
        schedule = ExponentialDecay.from_config(params)

    else:
        raise Exception("Unknown type of learning schedule: {}".format(type))
    return schedule


def optimizer(type, params,show_schedule=False):
    if "learning_rate" in params and isinstance(params["learning_rate"], dict):
        params_lr = params["learning_rate"]
        params["learning_rate"] = create_schedule(params_lr["type"], params_lr["params"])
        if show_schedule:
            show_schedule(params_lr["type"], params_lr["params"])
    if type == "adam":
        opti = tf.keras.optimizers.Adam(**params)
    elif type == "rms":
        opti = tf.keras.optimizers.RMSprop(**params)
    else:
        raise Exception()
    return opti


def create_model(config, action_size, obs_size, plot_model=True):
    s = tf.keras.layers.Input(shape=[obs_size, ], name='s')
    a = tf.keras.layers.Input(shape=[action_size, ], name='a')
    s_ = tf.keras.layers.Input(shape=[obs_size, ], name='s_')
    r_ = tf.keras.layers.Input(shape=[1, ], name='r_')
    inputs = [s, a]

    if config["type"] == BAYESIAN_CLASSIFIER:
        prior_params = config["prior_params"]

        kl_weight = config["kl_weight"]

    if config["model_key"] == "reward":
        inputs.append(r_)
    elif config["model_key"] == "dynamics":
        inputs.append(s_)
    else:
        raise Exception()
    x = tf.keras.layers.concatenate(inputs)

    for i_layer, layer in enumerate(config["hidden_layers"]):
        DV_or_D, layer = layer
        if DV_or_D == "Dense":
            l = tf.keras.layers.Dense(layer, activation=config["activation"])
            x = l(x)
            if config["batch_norm"]:
                x = tf.keras.layers.BatchNormalization()(x)
            if "dropout" in config:
                if config["dropout"] > 0:
                    x = tf.keras.layers.Dropout(config["dropout"])(x)
        elif DV_or_D == "DenseVariational":
            l = DenseVariational(units=layer,
                                 activation=config["activation"],
                                 kl_weight=kl_weight,
                                 prior_sigma_1=prior_params["prior_sigma_1"],
                                 prior_sigma_2=prior_params["prior_sigma_2"],
                                 prior_pi=prior_params["prior_pi"]
                                 )
            # init_sigma=config["init_sigma"])
            x = l(x)
        else:
            raise Exception()

    out_size = config["e_size"]
    if config["output_layers"] == "Dense":
        source_classification = tf.keras.layers.Dense(out_size, activation="softmax")(x)
    elif config["output_layers"] == "DenseVariational":
        source_classification = DenseVariational(
            units=out_size,
            activation="softmax",
            kl_weight=kl_weight,
            prior_sigma_1=prior_params["prior_sigma_1"],
            prior_sigma_2=prior_params["prior_sigma_2"],
            prior_pi=prior_params["prior_pi"]
        )(x)
    else:
        raise Exception()
    model = tf.keras.Model(inputs=inputs, outputs=source_classification)
    if plot_model:
        tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file="classifier.png")
    return model


def create_indices_for_factorize(e):
    rangee = tf.expand_dims(tf.range(0, e.shape[0], delta=1, dtype=tf.int64, name='range'), axis=1)
    e_real = tf.expand_dims(e, axis=1)
    indices = tf.concat((rangee, e_real), axis=1)
    return indices


def create_sparse_E(e, gan_config):
    indices = create_indices_for_factorize(e)
    values = tf.constant(1, shape=(e.shape[0],), dtype=tf.float32)
    dense_shape = (e.shape[0], gan_config["e_size"])
    E = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape
    )
    return tf.sparse.reorder(E)


cross_entropy = tf.keras.losses.CategoricalCrossentropy(reduction=ReductionV2.SUM_OVER_BATCH_SIZE)


def loss_function(e, model_ouput, config):
    output_label = tf.sparse.to_dense(create_sparse_E(e, config))

    if config["type"] == CLASSIFIER:
        loss = cross_entropy(output_label, model_ouput)
    elif config["type"] == BAYESIAN_CLASSIFIER:
        # if "use_keras_loss_for_BNN" in config and config["use_keras_loss_for_BNN"]:
        loss = cross_entropy(output_label, model_ouput)
        # else:
        #     loss = DenseVariational.cross_entropy_negative_log_likelihood(clip=config["clip"])(output_label,model_ouput)
    else:
        raise Exception("unkown type {}".format(config["type"]))

    info = {
        "loss": loss.numpy(),
    }
    return loss, info


def train(train_ds, test_ds, model, optimiser, config, log_folder, stop,
          interval_eval_epoch, callback=None, verbose=True):
    if config["batch_size"] == "max":
        batch_size = len(list(train_ds))
    else:
        batch_size = int(config["batch_size"])
    minibatches = train_ds.batch(batch_size)

    all_train_samples = train_ds.batch(len(list(train_ds)))
    all_tests_samples = test_ds.batch(len(list(test_ds)))

    plots, writers = create_plots_and_writers(log_folder)
    done = False
    epoch = 0
    while not done:
        done = perform_an_epoch(plots=plots, writers=writers, epoch=epoch,
                                minibatches=minibatches,
                                model=model, optimiser=optimiser,
                                config=config, stop=stop, callback=callback,
                                verbose=verbose, interval_eval_epoch=interval_eval_epoch,
                                all_tests_samples=all_tests_samples,
                                all_train_samples=all_train_samples)
        epoch += 1


def create_plots_and_writers(log_folder):
    plots = {
        "testing_loss": "training & testing",
        "debug_loss": "training & testing",
        "training_loss": "training & testing",
        "loss (minibatches)": "training & testing",
    }
    writers = {}
    for metric, _ in plots.items():
        writers[metric] = tf.summary.create_file_writer(str(log_folder / metric))

    writers["weights"] = tf.summary.create_file_writer(str(log_folder / "weights"))
    # writers["loss (minibatches)"] = tf.summary.create_file_writer(str(log_folder / "loss (minibatches)"))
    return plots, writers


def gradient_step(s, a, r_, s_, e, optimiser, NN, config):
    a = tf.cast(a, tf.float32)
    s = tf.cast(s, tf.float32)
    s_ = tf.cast(s_, tf.float32)
    r_ = tf.cast(r_, tf.float32)
    e = tf.cast(e, tf.int64)
    if config["model_key"] == "reward":
        real_out = r_
        real_out = tf.reshape(real_out, (-1, 1))

    elif config["model_key"] == "dynamics":
        real_out = s_
    else:
        raise Exception()

    with tf.GradientTape() as tape:
        output = NN([s, a, real_out], training=True)
        loss, info = loss_function(
            e=e,
            model_ouput=output,
            config=config)
    gradients = tape.gradient(loss, NN.trainable_variables)
    optimiser.apply_gradients(zip(gradients, NN.trainable_variables))
    return loss


def perform_an_epoch(plots, writers,
                     epoch, minibatches,
                     model, optimiser,
                     config,
                     all_tests_samples,
                     all_train_samples,
                     interval_eval_epoch,
                     stop, callback=None, verbose=True,
                     ):
    done = False
    isnan = False

    for n_minibatch, (s, a, r_, s_, e, _) in minibatches.enumerate():
        loss = gradient_step(s, a, r_, s_,
                             e=e,
                             optimiser=optimiser,
                             NN=model,
                             config=config)
        if n_minibatch % 100 == 0:
            print("[EPOCH={}][MINIBATCH={}][STEP={}] loss={}".format(epoch, n_minibatch,
                                                                     n_minibatch + len(list(minibatches)) * epoch,
                                                                     loss))

        with writers["loss (minibatches)"].as_default():
            tf.summary.scalar("training & testing", data=loss, step=epoch)  # n_minibatch + len(minibatches) * epoch)

    info = {
        "training_loss": None,
        "testing_loss": None
    }

    if interval_eval_epoch is not None and epoch % interval_eval_epoch == 0:
        print("EPOCH={}".format(epoch))

        for ds, str_loss in [(all_train_samples, "training_loss"), (all_tests_samples, "testing_loss")]:
            for n_minibatch, (s, a, r_, s_, e, _) in enumerate(ds):
                if n_minibatch > 0:
                    raise Exception("Should be only one batch when evaluating the network")
                a = tf.cast(a, tf.float32)
                s = tf.cast(s, tf.float32)
                e = tf.cast(e, tf.int64)
                inputs = [s, a]

                if config["model_key"] == "reward":
                    r_ = tf.cast(r_, tf.float32)
                    r_or_s = r_
                    r_or_s = tf.reshape(r_or_s, (-1, 1))
                elif config["model_key"] == "dynamics":
                    s_ = tf.cast(s_, tf.float32)
                    r_or_s = s_
                else:
                    raise Exception()

                inputs.append(r_or_s)
                output = model(inputs, training="training" in config and config["training"])
                t_loss, _ = loss_function(e, output, config)
                info[str_loss] = t_loss.numpy()

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

        with writers["weights"].as_default():
            for v in model.trainable_variables:
                tf.summary.histogram(v.name, data=v, step=epoch)

    if callback:
        callback(epoch, verbose, info["training_loss"], info["testing_loss"])

    if "max_epochs" in stop:
        if epoch > stop["max_epochs"]:
            done = True

    return done
