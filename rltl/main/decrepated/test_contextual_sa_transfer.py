from rltl.main.gan import CLASSIC, SUPER_GAN, AGGLOMERATED_GAN
from rltl.main.policy import RandomPolicy
from rltl.main.print_cc import create_heatmap
from rltl.utils.experiment_state import ExperimentData

from rltl.utils.utils_os import makedirs
from rltl.utils.utils_rl import rollouts
import numpy as np
import tensorflow as tf


def run(C, gan_baselines):
    exp = ExperimentData(C.path, C).load()

    common_parameters = {**C["learn_gans"]}
    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]

    for id_baseline, config_baseline in baselines.items():
        if gan_baselines is None or (gan_baselines is not None and id_baseline in gan_baselines):
            print("[baseline={}]".format(id_baseline))
            # config_baseline = {**config_baseline, **common_parameters}
            config_baseline = recursive_dict_update(common_parameters, config_baseline)
            path = C.path / "cross_comparaison" / id_baseline
            makedirs(path)
            if config_baseline["type"] == CLASSIC:
                pass
            elif config_baseline["type"] == AGGLOMERATED_GAN:
                pass
            elif config_baseline["type"] == SUPER_GAN:
                run_super_gan(id_baseline, C, exp, config_baseline, path=path)


def run_super_gan(id_baseline, C, exp, config_baseline, path):
    baseline = exp.learn_gans_data.baselines[id_baseline]
    sources_labels = []
    for env_name, (env_creator, config) in exp.source_collection.envs.items():
        sources_labels.append(str(config))
    for collection in [exp.source_collection, exp.target_collection]:
        print(">>>>>>>>>>>>><<<<>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>><<<< {} >>>>>>>>>>>>>>>>>>>>".format(collection.env_prefix))
        print(">>>>>>>>>>>>><<<<>>>>>>>>>>>>>>>>>>>>")
        targets_labels = []
        super_D_dynamics = baseline.get_model("best", config_baseline["type"], "dynamics", "D", "all_envs")
        super_D_reward = baseline.get_model("best", config_baseline["type"], "reward", "D", "all_envs")
        super_G_dynamics = baseline.get_model("best", config_baseline["type"], "dynamics", "G", "all_envs")
        super_G_reward = baseline.get_model("best", config_baseline["type"], "reward", "G", "all_envs")
        do_reward = super_G_reward is not None  # "gan_reward" in C["gan_config"] and C["gan_config"]["gan_reward"]
        do_dynamics = super_G_dynamics is not None  # "gan_dynamics" in C["gan_config"] and C["gan_config"]["gan_dynamics"]
        accuracies_targets = []
        accuracies_targets_r = []
        for i_env, (env_name, (env_creator, config)) in enumerate(collection.envs.items()):
            print("============================")
            print(config)
            print("============================")
            targets_labels.append(str(config))
            test_env = env_creator()
            # print("env.horizon={}".format(test_env.horizon))

            m, _ = rollouts(env_creator,
                            RandomPolicy(action_space=test_env.action_space),
                            num_episodes=C["cross_comparaison"]["nb_episodes"],
                            init_samples=None,
                            onehot_action=C["onehot_action"],
                            multiprocessing=False)

            source_samples = m.memory  # m.sample(C["cross_comparaison"]["nb_samples"])
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
                if do_dynamics:

                    accuracy_target = super_D_dynamics([s, a, s_])
                    if config_baseline["factorize_fake_output"]:
                        accuracy_target, _ = accuracy_target
                if do_reward:
                    r_new = tf.reshape(r_, (-1, 1))
                    accuracy_target_r = super_D_reward([s, a, r_new])
                    if config_baseline["factorize_fake_output"]:
                        accuracy_target_r, _ = accuracy_target_r
            if do_dynamics:
                accuracies_targets.append(np.mean(accuracy_target.numpy(), axis=0))
            if do_reward:
                accuracies_targets_r.append(np.mean(accuracy_target_r.numpy(), axis=0))
        if do_dynamics:
            print("acc sources vs sources (dynamics")
            create_latex_tab(accuracies_targets)
            create_heatmap("super_gan (dynamics, collection={})".format(collection.env_prefix),
                           accuracies_targets, sources_labels,
                           targets_labels, path=path / "dynamics", gan_config=config_baseline, show_plot=C.show_plots)
        if do_reward:
            print("acc sources vs sources (reward)")
            create_latex_tab(accuracies_targets_r)
            create_heatmap("super_gan (reward, collection={})".format(collection.env_prefix),
                           accuracies_targets_r, sources_labels,
                           targets_labels, path=path / "reward", gan_config=config_baseline, show_plot=C.show_plots)


def create_latex_tab(accs):
    strr = ""
    strr += "\\begin{center}\n"
    strr += "\\begin{tabular}{ " + " c " * len(accs[0]) + "}\n"
    for acc in accs:
        strr += (" & ".join(["{:.2f}".format(x) for x in acc]) + "\\\\" + "\n")
    strr = strr[:-3] + "\n"
    strr += "\\end{tabular}\n"
    strr += "\\end{center}"
    print(strr)
