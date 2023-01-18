from rltl.main.classifier import *
from rltl.main.learn_generative_model import get_gym_size, create_ds
from rltl.utils.utils_os import list_checker, override_dictionary
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.replay_memory import Memory
from rltl.utils.transition import HyperGanTransition
from rltl.utils.utils_os import empty_directory, makedirs, MyPool, save_object
import pathlib
from rltl.utils.utils_os import empty_directory


def run(C, classifier_baselines):
    common_parameters = {**C["learn_classifiers"]}
    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]

    args_pool = []
    for id_baseline, config_baseline in baselines.items():
        if list_checker(classifier_baselines, id_baseline):
            config = override_dictionary(common_parameters, config_baseline)
            args_pool.append((id_baseline, C, config))

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


def run_baseline(id_baseline, C, config_baseline):
    exp = ExperimentData(C.path, C).load(dont_load_models=True)
    print(">>>>>>>>>>>>>>>>>>>><<  [baseline={}] >>>>>>>>>>>>>>><<<<<<<<<<<<<<<".format(id_baseline))
    path = C.path / "learn_classifiers" / id_baseline
    makedirs(path)
    baseline = exp.learn_classifiers_data.new_baseline(id_baseline)
    config = config_baseline
    buffer_size_by_source_env = config["buffer_size_by_source_env"]

    verbose = C["learn_classifiers"]["verbose"]
    config["e_size"] = len(exp.envs_source)
    models_keys = []

    if "dynamics" in config:
        models_keys.append("dynamics")

    if "reward" in config:
        models_keys.append("reward")

    if not models_keys:
        raise Exception("you must specify at least 'reward' or 'dynamics' as model key")

    m = Memory(class_transition=HyperGanTransition)
    for i_env, (name, exp_env) in enumerate(exp.envs_source.items()):
        for sample in exp_env.memory.memory:
            m.push(sample.s, sample.a, sample.r_, sample.s_, i_env, -1)
    train_dataset, test_dataset = create_ds(
        obs_to_np=C["obs_to_np"],
        samples=m.sample(buffer_size_by_source_env * len(exp.envs_source)),
        ratio_train_to_test=C["learn_classifiers"]["ratio_train_to_test"])

    env_name, _ = next(iter(exp.envs_source.items()))
    from rltl.utils.registry import R
    env_creator, env_config = R.get_env(env_name)
    test_env = env_creator()
    action_size = test_env.action_space.n if C["onehot_action"] else get_gym_size(C,
        test_env.action_space)
    for model_key in models_keys:
        empty_directory(str(path / config["type"]))
        log_folder = path / config["type"] / "tf_log" / model_key

        print("logs folder: tensorboard --logdir={}".format(str(pathlib.Path().absolute() / log_folder)))
        if C["general"]["run_tensorboard"]:
            import os
            os.system("tensorboard --logdir={} &".format(str(pathlib.Path().absolute() / log_folder)))
        # recreate the gan_config for the rewar or dynamic (all of this is so ugly, I am ashamed)
        copy_config = {**config}

        copy_config["model_key"] = model_key

        for t in models_keys:
            del copy_config[t]
        copy_config = override_dictionary(copy_config, config[model_key])
        # copy_config = {**copy_config, **config[model_key]}

        if copy_config["type"] == BAYESIAN_CLASSIFIER:
            auto = False
            if "kl_weight" in config:
                kl_w = copy_config["kl_weight"]
                if kl_w == "auto":
                    auto = True
                else:
                    pass
            else:
                auto = True
            if auto:
                copy_config["kl_weight"] = 1. / len(list(train_dataset))  # 1 / nb_batches / sample_per_batch

        model = create_model(action_size=action_size,
                             obs_size=get_gym_size(C,test_env.observation_space),
                             config=copy_config)

        model_best = create_model(action_size=action_size,
                                  obs_size=get_gym_size(C,test_env.observation_space),
                                  config=copy_config)

        baseline.set_model(model, "current", config["type"], model_key, "classifier")
        baseline.set_model(model_best, "best", config["type"], model_key, "classifier")

        optimiser = optimizer(**copy_config["optimizer"], show_schedule=copy_config["show_schedule"])

        best_testing_loss = [np.inf]

        def save_callback(epoch, verbose, training_loss, testing_loss, local=best_testing_loss):
            # if config_baseline["interval_eval_epoch"] is not None and epoch % config_baseline["interval_eval_epoch"] == 0:
            if training_loss is not None and testing_loss is not None:
                print("training loss={} testing loss={}".format(training_loss, testing_loss))
                if testing_loss < local[-1]:
                    model_best.set_weights(model.get_weights())
                    local.append(testing_loss)
                    print(local)
            if epoch > 0 and epoch % copy_config["interval_save_model"] == 0:
                baseline.save()

        train(
            optimiser=optimiser,
            train_ds=train_dataset,
            model=model,
            test_ds=test_dataset,
            config=copy_config,
            log_folder=log_folder,
            verbose=verbose,
            stop=copy_config["stop"],
            callback=save_callback,
            interval_eval_epoch=copy_config["interval_eval_epoch"])

        baseline.save()
