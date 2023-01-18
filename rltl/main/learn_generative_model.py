from gym.spaces import Discrete, Box

from rltl.envs.observation_to_numpy import obs_to_np_factory
from rltl.main import inferences
from rltl.main.classifier import optimizer
from rltl.main.gan import *
from rltl.main.cvae import *
from rltl.main.spider import SPIDER, DenseSpider
from rltl.utils.replay_memory import Memory
from rltl.utils.utils_os import list_checker, override_dictionary
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.transition import HyperGanTransition
from rltl.utils.utils_os import makedirs, MyPool, save_object, empty_directory
import pathlib
import random
import tensorflow as tf


def create_ds(samples, ratio_train_to_test=0.7, obs_to_np="identity"):
    random.shuffle(samples)
    train_sample = samples[:int(ratio_train_to_test * len(samples))]
    obs_to_np = obs_to_np_factory(obs_to_np)
    S, A, R_, S_, E, C = ([obs_to_np(x.s) for x in train_sample],
                          [x.a for x in train_sample],
                          [x.r_ for x in train_sample],
                          [obs_to_np(x.s_) for x in train_sample],
                          [x.e for x in train_sample],
                          [x.c for x in train_sample])
    train_dataset = tf.data.Dataset.from_tensor_slices((S, A, R_, S_, E, C))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_sample))
    # if batch_size == "max":
    #     train_dataset = train_dataset.batch(len(train_sample))
    # else:
    #     train_dataset = train_dataset.batch(batch_size)

    if ratio_train_to_test < 1.:
        test_sample = samples[int(ratio_train_to_test * len(samples)):]
        S_test, A_test, R__test, S__test, E_test, C_test = ([obs_to_np(x.s) for x in test_sample],
                                                            [x.a for x in test_sample],
                                                            [x.r_ for x in test_sample],
                                                            [obs_to_np(x.s_) for x in test_sample],
                                                            [x.e for x in test_sample],
                                                            [x.c for x in test_sample])

        test_dataset = tf.data.Dataset.from_tensor_slices((S_test, A_test, R__test, S__test, E_test, C_test))
        test_dataset = test_dataset.shuffle(buffer_size=len(test_sample))
        # if batch_size == "max":
        #     test_dataset = test_dataset.batch(len(test_sample))
        # else:
        #     test_dataset = test_dataset.batch(batch_size)
    else:
        test_dataset = None
    return train_dataset, test_dataset


def get_gym_size(C, space):
    if isinstance(space, Discrete):
        return 1
    if isinstance(space, Box):
        return space.shape[0]

    o = space.sample()

    f = obs_to_np_factory(C["obs_to_np"])
    o = f(o)

    return len(o)


def run(C, gan_baselines, gan_single_envs):
    common_parameters = {**C["learn_gans"]}
    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]

    args_pool = []
    todo = 0
    for id_baseline, config_baseline in baselines.items():
        if list_checker(gan_baselines, id_baseline):
            # args_pool.append((id_baseline, C, {**config_baseline, **common_parameters}, gan_single_envs))
            config = override_dictionary(common_parameters, config_baseline)
            args_pool.append((id_baseline, C, config, gan_single_envs))
            todo += 1

    if todo == 0:
        raise Exception("Nothing to run, with {} and {}".format(gan_baselines, baselines.keys()))
    else:
        print("Running {}  baselines ({} and {})".format(todo, gan_baselines, baselines.keys()))
    if C.multiprocessing:
        # from ray.util.multiprocessing import Pool
        # ray.init(num_cpus=4,num_gpus=1)
        # from concurrent.futures import ProcessPoolExecutor as Pool #to get childrens
        # from multiprocessing import Pool
        n_processes = len(args_pool)
        print("Running {} processes (baselines) in parralle".format(n_processes))
        with MyPool(n_processes) as p:
            p.starmap(run_baseline, args_pool)
    else:
        for argg in args_pool:
            run_baseline(*argg)


def run_baseline(id_baseline, C, config_baseline, gan_single_envs):
    run_tensorboard = C["general"]["run_tensorboard"]
    # print("CUDA_VISIBLE_DEVICES={}".format(os.environ['CUDA_VISIBLE_DEVICES']))
    dont_load_models = True
    if config_baseline["type"] in [HYPER_GAN, HYPER_NN]:
        dont_load_models = False

    exp = ExperimentData(C.path, C).load(dont_load_models=dont_load_models)
    print(">>>>>>>>>>>>>>>>>>>><<  [baseline={}] >>>>>>>>>>>>>>><<<<<<<<<<<<<<<".format(id_baseline))
    path = C.path / "learn_gans" / id_baseline
    makedirs(path)
    if config_baseline["type"] in [HYPER_GAN, HYPER_NN]:
        if "learn_with_onehot_context" in config_baseline and config_baseline["learn_with_onehot_context"]:
            pass
        else:
            baseline_classifier = exp.learn_classifiers_data.baselines[config_baseline["classifier_baseline"]]
            model_dynamics = baseline_classifier.get_model("best", "classifier", "dynamics", "classifier", "all_envs")
    # baseline = exp.learn_gans_data.baselines[id_baseline]
    baseline = exp.learn_gans_data.new_baseline(id_baseline)
    gan_config = config_baseline
    buffer_size_by_source_env = gan_config["buffer_size_by_source_env"]
    lambda_ = gan_config["lambda_"]
    D_n_updates = gan_config["D_n_updates"]
    G_n_updates = gan_config["G_n_updates"]

    verbose = C["learn_gans"]["verbose"]
    gan_config["e_size"] = len(exp.envs_source)
    models_keys = []

    if "dynamics" in gan_config:
        models_keys.append("dynamics")

    if "reward" in gan_config:
        models_keys.append("reward")

    if gan_config["type"] in [SUPER_GAN, AGGLOMERATED_GAN, HYPER_GAN, HYPER_NN, HYPER_VAE, SPIDER]:

        # if gan_config["type"] == HYPER_GAN:
        class_transition = HyperGanTransition

        # else:
        #     class_transition = TransferTransition
        m = Memory(class_transition=class_transition)
        for i_env, (name, exp_env) in enumerate(exp.envs_source.items()):
            S, A, R_, S_ = ([x.s for x in exp_env.memory.memory],
                            [x.a for x in exp_env.memory.memory],
                            [x.r_ for x in exp_env.memory.memory],
                            [x.s_ for x in exp_env.memory.memory])
            steps = tf.data.Dataset.from_tensor_slices((S, A, R_, S_))
            steps = steps.batch(len(S))
            for (s, a, r_, s_) in steps:
                a = tf.cast(a, tf.float32)
                s = tf.cast(s, tf.float32)
                s_ = tf.cast(s_, tf.float32)
                r_ = tf.cast(r_, tf.float32)
            if gan_config["type"] in [HYPER_GAN, HYPER_NN, HYPER_VAE]:
                if "learn_with_onehot_context" in gan_config and gan_config["learn_with_onehot_context"]:
                    c = np.zeros((len(S), len(exp.envs_source),))
                    c[:, i_env] = 1.
                else:
                    c = model_dynamics([s, a, s_])
            else:
                c = [np.nan] * len(exp.envs_source)
            # for sample in exp_env.memory.memory:
            for si, ai, ri_, si_, ci in zip(s, a, r_, s_, c):
                # m.push(sample.s, sample.a, sample.r_, sample.s_, i_env)
                m.push(si, ai, ri_, si_, i_env, ci)
        train_ds, test_ds = create_ds(
            m.sample(buffer_size_by_source_env * len(exp.envs_source)),
            ratio_train_to_test=C["learn_gans"]["ratio_train_to_test"],
            obs_to_np=C["obs_to_np"]
        )
        env_name, _ = next(iter(exp.envs_source.items()))
        from rltl.utils.registry import R
        env_creator, env_config = R.get_env(env_name)
        test_env = env_creator()
        action_size = test_env.action_space.n if C["onehot_action"] else get_gym_size(C,
                                                                                      test_env.action_space)
        obs_size = get_gym_size(C, test_env.observation_space)
        for model_key in models_keys:
            if model_key == "reward":
                real_out_str = "r_"
            elif model_key == "dynamics":
                real_out_str = "s_"
            elif model_key == "done":
                real_out_str = "done"
            else:
                raise Exception()

            empty_directory(str(path / gan_config["type"]))
            log_folder = path / gan_config["type"] / "tf_log" / model_key

            print("logs folder: tensorboard --logdir={}".format(str(pathlib.Path().absolute() / log_folder)))
            if run_tensorboard:
                import os
                os.system("tensorboard --logdir={} &".format(str(pathlib.Path().absolute() / log_folder)))
            # recreate the gan_config for the rewar or dynamic (all of this is so ugly, I am ashamed)
            copy_config = {**gan_config}

            if gan_config["type"] == AGGLOMERATED_GAN:
                copy_config["type"] = CLASSIC
            copy_config["model_key"] = model_key

            for t in models_keys:
                del copy_config[t]
            # copy_config = {**copy_config, **gan_config[model_key]}
            copy_config = override_dictionary(copy_config, gan_config[model_key])
            generative_type = copy_config["generative_type"]
            show_schedule = "show_schedule" in copy_config and copy_config["show_schedule"]
            if generative_type == "GAN":
                if gan_config["type"] != HYPER_NN:
                    D = discriminator(action_size=action_size,
                                      obs_size=get_gym_size(C, test_env.observation_space),
                                      gan_config=copy_config,
                                      plot_model_path=path)
                    D_best = discriminator(action_size=action_size,
                                           obs_size=get_gym_size(C, test_env.observation_space),
                                           gan_config=copy_config)
                    baseline.set_model(D, "current", gan_config["type"], model_key, "D")
                    baseline.set_model(D_best, "best", gan_config["type"], model_key, "D")
                    D_opti = optimizer(**copy_config["D_optimizer"], show_schedule=show_schedule)
                else:
                    D = None
                    D_opti = None
                G = generator(action_size=action_size,
                              obs_size=obs_size,
                              gan_config=copy_config,
                              plot_model_path=path)

                G_best = generator(action_size=action_size,
                                   obs_size=obs_size,
                                   gan_config=copy_config)

                baseline.set_model(G, "current", gan_config["type"], model_key, "G")
                baseline.set_model(G_best, "best", gan_config["type"], model_key, "G")
                G_opti = optimizer(**copy_config["G_optimizer"], show_schedule=show_schedule)
            elif generative_type == "CVAE":
                cvae = DenseCVAE(
                    action_size, obs_size,
                    plot_model_path=path,
                    encoder_hiddens=copy_config["encoder_hiddens"],
                    decoder_hiddens=copy_config["decoder_hiddens"],
                    encoder_activation=copy_config["encoder_activation"],
                    decoder_activation=copy_config["decoder_activation"],
                    config_cvae=copy_config
                )
                cvae_best = DenseCVAE(
                    action_size, obs_size,
                    encoder_hiddens=copy_config["encoder_hiddens"],
                    decoder_hiddens=copy_config["decoder_hiddens"],
                    encoder_activation=copy_config["encoder_activation"],
                    decoder_activation=copy_config["decoder_activation"],
                    config_cvae=copy_config)
                baseline.set_model(cvae.decoder, "current", gan_config["type"], model_key, "decoder", name)
                baseline.set_model(cvae_best.decoder, "best", gan_config["type"], model_key, "decoder", name)
                opti_cvae = optimizer(**copy_config["CVAE_optimizer"], show_schedule=copy_config["show_schedule"])
            elif generative_type == "spider":
                spider = DenseSpider(
                    action_size, obs_size,
                    plot_model_path=path,
                    encoder_hiddens=copy_config["encoder_hiddens"],
                    decoder_hiddens=copy_config["decoder_hiddens"],
                    encoder_activation=copy_config["encoder_activation"],
                    decoder_activation=copy_config["decoder_activation"],
                    config=copy_config,
                    source_tasks_number=len(exp.envs_source)
                )
                spider_best = DenseSpider(
                    action_size, obs_size,
                    encoder_hiddens=copy_config["encoder_hiddens"],
                    decoder_hiddens=copy_config["decoder_hiddens"],
                    encoder_activation=copy_config["encoder_activation"],
                    decoder_activation=copy_config["decoder_activation"],
                    config=copy_config,
                    source_tasks_number=len(exp.envs_source)
                )
                baseline.set_model(spider.decoder, "current", gan_config["type"], model_key, "decoder", name)
                baseline.set_model(spider_best.decoder, "best", gan_config["type"], model_key, "decoder", name)
                opti_cvae = optimizer(**copy_config["spider_optimizer"], show_schedule=copy_config["show_schedule"])

            best_testing_loss = [np.inf]

            def save_callback(epoch, verbose, training_loss, testing_loss, local=best_testing_loss):
                if training_loss is not None and testing_loss is not None:
                    print("training loss={} testing loss={}".format(training_loss, testing_loss))
                    if testing_loss < local[-1] and not np.isnan(testing_loss):
                        if generative_type == "GAN":
                            if gan_config["type"] != HYPER_NN:
                                D_best.set_weights(D.get_weights())
                            G_best.set_weights(G.get_weights())
                        elif generative_type == "CVAE":
                            cvae_best.set_weights(cvae.get_weights())
                        elif generative_type == "spider":
                            raise Exception("oops")
                            # cvae_best.set_weights(cvae.get_weights())
                        local.append(testing_loss)
                        print("------>>>>>>>>> setting best weight with loss = {}".format(testing_loss))
                if epoch > 0 and epoch % config_baseline["interval_save_model"] == 0:
                    baseline.save()
                if epoch > 0 and epoch % config_baseline["interval_inferences"] == 0:
                    old_type = C["inferences"]["context_type"]
                    C["inferences"]["context_type"] = "bypass"
                    inferences.run(C, [id_baseline], [name], path / gan_config["type"], "epoch={}_".format(epoch))
                    C["inferences"]["context_type"] = old_type

            interval_eval_epoch = copy_config["interval_eval_epoch"]
            if generative_type == "GAN":
                train(
                    real_out_str=real_out_str,
                    obs_size=obs_size,
                    train_ds=train_ds,
                    test_ds=test_ds,
                    lambda_=lambda_,
                    G=G,
                    D=D,
                    G_opti=G_opti,
                    D_opti=D_opti,
                    D_n_updates=D_n_updates,
                    G_n_updates=G_n_updates,
                    gan_config=copy_config,
                    log_folder=log_folder,
                    verbose=verbose,
                    stop=copy_config["stop"],
                    callback=save_callback,
                    interval_eval_epoch=interval_eval_epoch)
            elif generative_type == "CVAE":
                cvae.train(
                    cvae_config=copy_config,
                    real_out_str=real_out_str,
                    log_folder=log_folder,
                    callback=save_callback,
                    verbose=verbose,
                    interval_eval_epoch=interval_eval_epoch,
                    obs_size=obs_size,
                    optimizer=opti_cvae,
                    train_ds=train_ds,
                    test_ds=test_ds,
                    stop=copy_config["stop"])
            elif generative_type == "spider":
                spider.train()

        baseline.save()
    elif gan_config["type"] == CLASSIC:

        models_keys = []

        if "dynamics" in gan_config:
            models_keys.append("dynamics")

        if "reward" in gan_config:
            models_keys.append("reward")

        todo = [(exp.envs_source)]
        if "learn_targets" in config_baseline and config_baseline["learn_targets"] == True:
            todo.append(exp.envs_target)
        args_pool = []
        for coll in todo:
            todo_env = 0
            for name, exp_env in coll.items():
                if list_checker(gan_single_envs, name):
                    # if gan_single_envs is not None and not (name in gan_single_envs):
                    #     add_it = False
                    # if add_it:
                    for model_key in models_keys:
                        args_pool.append((
                            id_baseline,
                            C,
                            run_tensorboard,
                            buffer_size_by_source_env,
                            exp_env.memory.memory,
                            model_key,
                            verbose,
                            D_n_updates,
                            G_n_updates,
                            C["onehot_action"],
                            baseline,
                            path,
                            config_baseline,
                            name,
                            models_keys,
                            gan_config,
                            lambda_,
                            C["learn_gans"]["ratio_train_to_test"]
                        ))
                todo_env += 1
            if todo_env == 0:
                raise Exception("no env to learn ({}, {})".format(coll.keys(), gan_single_envs))
        if C.multiprocessing:
            from multiprocessing import Pool
            # from concurrent.futures import ProcessPoolExecutor as Pool
            # from ray.util.multiprocessing import Pool

            n_processes = len(args_pool)
            print("Running {} processes (envs and reward or dynamics) in parralle".format(n_processes))
            with Pool(n_processes) as p:
                p.starmap(learn_simple_gan_for_single_env, args_pool)
        else:
            for argg in args_pool:
                learn_simple_gan_for_single_env(*argg)

    else:
        raise Exception("unknown gan type={}".format(gan_config["type"]))


def learn_simple_gan_for_single_env(
        id_baseline,
        C,
        run_tensorboard,
        buffer_size_by_source_env,
        memory_from_create_dataset,
        model_key,
        verbose,
        D_n_updates,
        G_n_updates,
        use_one_hot,
        baseline,
        path,
        config_baseline,
        name,
        models_keys,
        gan_config,
        lambda_,
        ratio_train_to_test):
    print("learning generative model for single env={}".format(name))
    memory = Memory(class_transition=HyperGanTransition)
    for sample in memory_from_create_dataset:
        memory.push(sample.s, sample.a, sample.r_, sample.s_, -1, -1)
    samples = memory.sample(buffer_size_by_source_env)
    train_ds, test_ds = create_ds(samples, ratio_train_to_test=ratio_train_to_test, obs_to_np=C["obs_to_np"])

    best_testing_loss = [np.inf]

    if model_key == "reward":
        real_out_str = "r_"
    elif model_key == "dynamics":
        real_out_str = "s_"
    elif model_key == "done":
        real_out_str = "done"
    else:
        raise Exception()
    log_folder = path / config_baseline["type"] / name / "tf_log" / model_key
    empty_directory(str(path / config_baseline["type"] / name))
    save_object(config_baseline, str(path / config_baseline["type"] / name / "config.yaml"))
    print("logs folder: tensorboard --logdir={}".format(str(pathlib.Path().absolute() / log_folder)))
    if run_tensorboard:
        import os
        import subprocess
        # os.system()
        subprocess.run('bash -c "source activate rltl;' + 'tensorboard --logdir={} "&'
                       .format(str(pathlib.Path().absolute() / log_folder)), shell=True)
    from rltl.utils.registry import R
    env_creator, env_config = R.get_env(name)
    test_env = env_creator()
    action_size = test_env.action_space.n if use_one_hot else get_gym_size(C,
                                                                           test_env.action_space)

    # recreate the gan_config for the rewar or dynamic (all of this is so ugly, I am ashamed)
    copy_config = {**gan_config}

    if gan_config["type"] == AGGLOMERATED_GAN:
        raise Exception("Impossibruuu")
        copy_config["type"] = CLASSIC

    copy_config["model_key"] = model_key

    for t in models_keys:
        del copy_config[t]
    # copy_config = {**copy_config, **gan_config[model_key]}
    copy_config = override_dictionary(copy_config, gan_config[model_key])

    generative_type = copy_config["generative_type"]
    obs_size = get_gym_size(C, test_env.observation_space)
    if generative_type == "GAN":
        D = discriminator(action_size=action_size,
                          obs_size=obs_size,
                          gan_config=copy_config,
                          plot_model_path=path)
        G = generator(action_size=action_size,
                      obs_size=obs_size,
                      gan_config=copy_config,
                      plot_model_path=path)
        D_best = discriminator(action_size=action_size,
                               obs_size=obs_size,
                               gan_config=copy_config)
        G_best = generator(action_size=action_size,
                           obs_size=obs_size,
                           gan_config=copy_config)
        baseline.set_model(G, "current", gan_config["type"], model_key, "G", name)
        baseline.set_model(G_best, "best", gan_config["type"], model_key, "G", name)
        baseline.set_model(D, "current", gan_config["type"], model_key, "D", name)
        baseline.set_model(D_best, "best", gan_config["type"], model_key, "D", name)

        G_opti = optimizer(**copy_config["G_optimizer"])
        D_opti = optimizer(**copy_config["D_optimizer"])
    elif generative_type == "CVAE":
        sigmoid_output = "sigmoid_output" in copy_config and copy_config["sigmoid_output"]
        cvae = DenseCVAE(
            action_size, obs_size,
            plot_model_path=path,
            encoder_hiddens=copy_config["encoder_hiddens"],
            decoder_hiddens=copy_config["decoder_hiddens"],
            encoder_activation=copy_config["encoder_activation"],
            decoder_activation=copy_config["decoder_activation"],
            config_cvae=copy_config
        )
        cvae_best = DenseCVAE(
            action_size, obs_size,
            encoder_hiddens=copy_config["encoder_hiddens"],
            decoder_hiddens=copy_config["decoder_hiddens"],
            encoder_activation=copy_config["encoder_activation"],
            decoder_activation=copy_config["decoder_activation"],
            config_cvae=copy_config)
        baseline.set_model(cvae.decoder, "current", gan_config["type"], model_key, "decoder", name)
        baseline.set_model(cvae_best.decoder, "best", gan_config["type"], model_key, "decoder", name)
        opti_cvae = optimizer(**copy_config["CVAE_optimizer"])
    else:
        raise Exception("Unknown generative_type: {}".format(generative_type))

    def save_callback(epoch, verbose, training_loss, testing_loss, local=best_testing_loss):
        if training_loss is not None and testing_loss is not None:
            if testing_loss < local[-1]:
                if generative_type == "GAN":
                    D_best.set_weights(D.get_weights())
                    G_best.set_weights(G.get_weights())
                elif generative_type == "CVAE":
                    cvae_best.set_weights(cvae.get_weights())
                local.append(testing_loss)
                print(local)
        if epoch > 0 and epoch % config_baseline["interval_save_model"] == 0:
            baseline.save()
        if epoch > 0 and epoch % config_baseline["interval_inferences"] == 0:
            inferences.run(C, [id_baseline], [name], path / gan_config["type"], "epoch={}_".format(epoch))

    interval_eval_epoch = copy_config["interval_eval_epoch"]
    if generative_type == "GAN":
        train(
            real_out_str=real_out_str,
            obs_size=obs_size,
            train_ds=train_ds,
            test_ds=test_ds,
            lambda_=lambda_,
            G=G,
            D=D,
            G_opti=G_opti,
            D_opti=D_opti,
            D_n_updates=D_n_updates,
            G_n_updates=G_n_updates,
            gan_config=copy_config,
            log_folder=log_folder,
            verbose=verbose,
            stop=copy_config["stop"],
            callback=save_callback,
            interval_eval_epoch=interval_eval_epoch)
    elif generative_type == "CVAE":
        cvae.train(
            cvae_config=copy_config,
            real_out_str=real_out_str,
            log_folder=log_folder,
            callback=save_callback,
            verbose=verbose,
            interval_eval_epoch=interval_eval_epoch,
            obs_size=obs_size,
            optimizer=opti_cvae,
            train_ds=train_ds,
            test_ds=test_ds,
            stop=copy_config["stop"])
    baseline.save()
