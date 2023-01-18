from rltl.main.classifier import BAYESIAN_CLASSIFIER
from rltl.main.cvae import HYPER_VAE, SamplerCVAE
from rltl.main.gan import CLASSIC, SUPER_GAN, AGGLOMERATED_GAN, HYPER_GAN, HYPER_NN
from rltl.utils.utils_os import list_checker, override_dictionary
from rltl.main.policy import RandomPolicy, SingleActionPolicy
from rltl.main.print_cc import create_heatmap
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.registry import R
from rltl.utils.replay_memory import Memory
from rltl.utils.transition import TransitionGym
from rltl.utils.utils_os import makedirs, save_object
from rltl.utils.utils_rl import exploration, change_env_state_gridworld, rollout, reverse_one_hot
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def run(C, bayesian_baselines, path_to_save=None):
    exp = ExperimentData(C.path, C).load(dont_load_models=True)

    common_parameters_classifier = {**C["learn_classifiers"]}
    baselines_classifiers = common_parameters_classifier["baselines"]

    del common_parameters_classifier["baselines"]
    path_to_save = C.path if path_to_save is None else path_to_save

    makedirs(str(path_to_save / "threshold_context_sigma"))
    all_sigmas = {}

    for i_env, (env_name, (env_creator, config)) in enumerate(exp.source_collection.envs.items()):
        m = exploration(
            env_creator=env_creator,
            change_env_init_state=change_env_state_gridworld,
            one_hot_action=C["onehot_action"],
            **C["threshold_context_sigma"]["exploration"])

        source_samples = m.memory
        S, A, R_, S_ = ([x.s for x in source_samples],
                        [x.a for x in source_samples],
                        [x.r_ for x in source_samples],
                        [x.s_ for x in source_samples])
        steps = tf.data.Dataset.from_tensor_slices((S, A, R_, S_))
        steps = steps.batch(len(S))

        for (s, a, r_, s_) in steps:
            a = tf.cast(a, tf.float32)
            s = tf.cast(s, tf.float32)
            s_ = tf.cast(s_, tf.float32)
            r_ = tf.cast(r_, tf.float32)

        actual_nb_of_baseline = 0
        for id_baseline, config_baseline in baselines_classifiers.items():
            print("id_baseline={}".format(id_baseline))
            if list_checker(bayesian_baselines, id_baseline):
                if id_baseline in exp.learn_classifiers_data.baselines:
                    exp.learn_classifiers_data.load_baseline(id_baseline)
                    baseline_classifier = exp.learn_classifiers_data.baselines[id_baseline]

                    context_model = baseline_classifier.get_model(
                        "best", "bayesian_classifier", "dynamics",
                        "classifier", "all_envs")
                    if context_model is not None:
                        # repeat inference and get mu and sigma
                        contexts = []
                        config_classifier = override_dictionary(common_parameters_classifier, config_baseline)
                        training = "training" in config_classifier and config_classifier["training"]
                        for _ in range(C["threshold_context_sigma"]["repeat_bayesian_inferences"]):
                            contexts.append(context_model([s, a, s_], training=training))
                        # mu_c = np.mean(contexts, axis=0)
                        sigma_c = np.std(contexts, axis=0)
                        if id_baseline not in all_sigmas:
                            all_sigmas[id_baseline] = sigma_c
                        else:
                            all_sigmas[id_baseline] = np.concatenate([all_sigmas[id_baseline], sigma_c])

                    else:
                        raise Exception("No context model")

                    actual_nb_of_baseline += 1
                else:
                    print("warning: baseline {} does not exist in the data".format(id_baseline))

    data = {}
    for id_baseline, sigmas in all_sigmas.items():
        print(sigmas)
        print(np.mean(sigmas))
        h = np.histogram(sigmas)
        print(h)
        import matplotlib.pyplot as plt
        plt.hist(sigmas)
        plt.title(id_baseline)
        plt.show()
        data[id_baseline] = {
            "quantile95": float(np.quantile(sigmas, 0.95)),
            "quantile75": float(np.quantile(sigmas, 0.75)),
            "quantile50": float(np.quantile(sigmas, 0.50)),
            "mean": float(np.mean(sigmas)),
            "max": float(np.max(sigmas))
        }
        print(data)

    save_object(data, str(path_to_save / "threshold_context_sigma" / "data.json"))
    # if actual_nb_of_baseline ==0:
    #     raise Exception("No Baseline processed {}, {}".format(bayesian_baselines))
