from rltl.envs.gridworld.world import World
from rltl.main.classifier import CLASSIFIER, BAYESIAN_CLASSIFIER
from rltl.main.gan import CLASSIC, SUPER_GAN, AGGLOMERATED_GAN
from rltl.main.policy import RandomPolicy
from rltl.main.print_cc import create_heatmap
from rltl.utils.experiment_state import ExperimentData

from rltl.utils.utils_os import makedirs, list_checker, override_dictionary
from rltl.utils.utils_rl import rollouts, change_env_state_gridworld, exploration
import numpy as np
import tensorflow as tf
import matplotlib as plt

"""
red is the plot of the ith dimension of the context vector
the corresponding ith task distribution is displayed in white
the distribution of the task for which we plot the context vector is in green
"""

colors = [
    np.array([1., 0., 0.]),  # red
    np.array([0., 1., 0.]),  # green
    np.array([0., 0., 1.]),  # blue
    np.array([1., 0., 1.]),  # pink
    np.array([1., 1., 0.]),
    np.array([0., 1., 1.]),
    np.array([.5, 0., 0.]),  # red
    np.array([0., .5, 0.]),  # green
    np.array([0., 0., .5]),  # blue
    np.array([.5, 0., .5]),  # pink
    np.array([.5, .5, 0.]),
    np.array([0., .5, .5]),
]

for _ in range(100):
    colors.append([0.5, 0.5, 0.5])


def run(C, classifier_baselines, envs_to_test):
    exp = ExperimentData(C.path, C).load(dont_load_models=True)

    common_parameters = {**C["learn_classifiers"]}
    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]

    for id_baseline, config_baseline in baselines.items():
        if list_checker(classifier_baselines, id_baseline):
            print("[baseline={}]".format(id_baseline))
            config_baseline = override_dictionary(common_parameters, config_baseline)
            path = C.path / "cross_comparaison" / "learn_classifiers" / id_baseline
            makedirs(path)
            run_super_gan(id_baseline, C, exp, config_baseline, path=path, envs_to_test=envs_to_test)


def run_super_gan(id_baseline, C, exp, config_baseline, path, envs_to_test):
    exp.learn_classifiers_data.load_baseline(id_baseline)
    baseline = exp.learn_classifiers_data.baselines[id_baseline]
    sources_trajs = {}
    targets_trajs = {}
    repeat_sources = C["visualize_classifiers"]["repeat_sources"]
    print("contructing samples ...")
    for trajs, collection in [(sources_trajs, exp.source_collection), (targets_trajs, exp.target_collection)]:
        for target_env_name, (env_creator, config) in collection.envs.items():
            test_env = env_creator()
            m = exploration(
                env_creator=env_creator,
                change_env_init_state=change_env_state_gridworld,
                one_hot_action=C["onehot_action"],
                type="grid",
                n_repeat=repeat_sources,
                deltas=C["visualize_classifiers"]["deltas_sources"],
                std=0.0)

            source_samples = m.memory
            S, A, R_, S_ = ([x.s for x in source_samples],
                            [x.a for x in source_samples],
                            [x.r_ for x in source_samples],
                            [x.s_ for x in source_samples])

            trajs[target_env_name] = (S, A, R_, S_)
            trajectories = []
            for i, (s, a, r_, s_) in enumerate(zip(S, A, R_, S_)):
                # print("s beofre: ", s)
                true_s = np.array(s)
                true_s_ = np.array(s_)
                if test_env.normalise_state:
                    true_s[0] = true_s[0] * test_env.w
                    true_s[1] = true_s[1] * test_env.h
                    true_s_[0] = true_s_[0] * test_env.w
                    true_s_[1] = true_s_[1] * test_env.h
                # print("s after", s)
                S[i] = true_s
                S_[i] = true_s_
                trajectories.append([(true_s, a, r_, true_s_, None, None)])

            w = World(e=test_env)
            w.draw_frame()
            w.draw_cases()
            w.draw_lattice()

            w.draw_trajectories(trajectories, rgba=(1, 1, 1, C["env_statistics"]["alpha"]), line_width=1)
            makedirs(C.path / "visualize_classifiers" / id_baseline)
            filename = target_env_name + "_ground_truth"
            w.save(str(C.path / "visualize_classifiers" / id_baseline / filename))
            if C.show_plots:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                img = mpimg.imread(
                    str(C.path / "visualize_classifiers" / id_baseline / (filename + ".png")))
                imgplot = plt.imshow(img)
                plt.show()
            # exit()
    if config_baseline["type"] in [CLASSIFIER, BAYESIAN_CLASSIFIER]:
        type_network = "classifier"
    else:
        type_network = "D"
    model_dynamics = baseline.get_model("best", config_baseline["type"], "dynamics", type_network, "all_envs")
    model_reward = baseline.get_model("best", config_baseline["type"], "reward", type_network, "all_envs")
    do_reward = model_reward is not None  # "gan_reward" in C["gan_config"] and C["gan_config"]["gan_reward"]
    do_dynamics = model_dynamics is not None  # "gan_dynamics" in C["gan_config"] and C["gan_config"]["gan_dynamics"]
    accuracies_targets = []
    accuracies_targets_r = []
    if not do_reward and not do_dynamics:
        print("[{}] Reward and Dynamics models are none".format(id_baseline))
        return None

    deltas = C["visualize_classifiers"]["deltas_context"]
    w_pixel = deltas[0] - deltas[0] / 10
    h_pixel = deltas[1] - deltas[1] / 10
    repeat_context = C["visualize_classifiers"]["repeat_context"]

    for collection in [exp.source_collection, exp.target_collection]:
        print(">>>>>>>>>>>>><<<<>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>><<<< {} >>>>>>>>>>>>>>>>>>>>".format(collection.env_prefix))
        print(">>>>>>>>>>>>><<<<>>>>>>>>>>>>>>>>>>>>")
        targets_labels = []

        for i_env, (target_env_name, (env_creator, config)) in enumerate(collection.envs.items()):
            if list_checker(envs_to_test, target_env_name):
                print("============================")
                print(config)
                print("============================")
                targets_labels.append(str(config))

                m = exploration(
                    env_creator=env_creator,
                    change_env_init_state=change_env_state_gridworld,
                    one_hot_action=C["onehot_action"],
                    type="grid",
                    n_repeat=repeat_context,
                    deltas=deltas,
                    std=0.0)

                source_samples = m.memory
                S, A, R_, S_ = ([x.s for x in source_samples],
                                [x.a for x in source_samples],
                                [x.r_ for x in source_samples],
                                [x.s_ for x in source_samples])
                steps = tf.data.Dataset.from_tensor_slices((S, A, R_, S_))
                steps = steps.batch(len(S))
                for i, (s, a, r_, s_) in enumerate(steps):
                    if i >= 1:
                        raise Exception()
                    a = tf.cast(a, tf.float32)
                    s = tf.cast(s, tf.float32)
                    s_ = tf.cast(s_, tf.float32)
                    r_ = tf.cast(r_, tf.float32)
                    if do_dynamics:
                        if config_baseline["type"] == BAYESIAN_CLASSIFIER:
                            repeat_bayesian_inference = C["visualize_classifiers"]["repeat_bayesian_inferences"]
                        else:
                            repeat_bayesian_inference = 1
                        accuracies = []
                        for _ in range(repeat_bayesian_inference):
                            mu_accuracy_target = model_dynamics([s, a, s_])
                            accuracies.append(mu_accuracy_target)
                        mu_accuracy_target = np.mean(accuracies, axis=0)
                        sigma_accuracy_target = np.std(accuracies, axis=0)
                # np.save(str(C.path / "visualize_classifiers" / id_baseline / "accuracies" / "mu_{}.npy".format(target_env_name)),mu_accuracy_target)
                # np.save(str(C.path / "visualize_classifiers" / id_baseline / "accuracies" / "sigma_{}.npy".format(target_env_name)),sigma_accuracy_target)
                test_env = env_creator()

                for i, (s, a, s_) in enumerate(zip(S, A, S_)):
                    true_s = np.array(s)  # need to copy
                    true_s_ = np.array(s_)
                    if test_env.normalise_state:
                        true_s[0] = true_s[0] * test_env.w
                        true_s[1] = true_s[1] * test_env.h
                        true_s_[0] = true_s_[0] * test_env.w
                        true_s_[1] = true_s_[1] * test_env.h
                    S[i] = true_s
                    S_[i] = true_s_
                if test_env.normalise_state:
                    h_p = h_pixel * test_env.h
                    w_p = w_pixel * test_env.w

                for metric, value, colr in [("mu", mu_accuracy_target, (1, 0, 0)),
                                            ("sigma", sigma_accuracy_target, (1, 1, 0))]:

                    if target_env_name in sources_trajs:
                        gt_trajs = sources_trajs[target_env_name]
                    elif target_env_name in targets_trajs:
                        gt_trajs = targets_trajs[target_env_name]
                    else:
                        raise Exception("impossible")

                    # drawing the plot for each context dimension
                    for ctx_dimension, (ctx_env_name, (_, _)) in enumerate(exp.source_collection.envs.items()):
                        w = World(test_env)
                        w.draw_frame()
                        w.draw_lattice()
                        w.draw_cases()
                        for i, (s, a, s_, c) in enumerate(zip(S, A, S_, value)):
                            w.draw_rectangle(
                                (s[0] - w_p / 2, s[1] - h_p / 2, s[0] + w_p / 2, s[1] + h_p / 2),
                                color=(colr[0], colr[1], colr[2], (1. / repeat_context) * c[ctx_dimension]))
                        for (s, a, r_, s_) in zip(*sources_trajs[ctx_env_name]):
                            w.draw_trajectory([(s, None, None, s_, None, None)],
                                              (0, 1, 0, (1 / repeat_sources)),  # * len(sources_trajs)),
                                              line_width=1)

                        for (s, a, r_, s_) in zip(*gt_trajs):
                            w.draw_trajectory([(s, None, None, s_, None, None)],
                                              (1, 1, 1, (1 / repeat_sources)),  # * len(sources_trajs)),
                                              line_width=1)
                        makedirs(C.path / "visualize_classifiers" / id_baseline)
                        filename = "target=" + target_env_name + "_(white)_ctx=" + ctx_env_name + "_(green)_" + metric
                        w.save(str(C.path / "visualize_classifiers" / id_baseline / filename))
                        if C.show_plots:
                            import matplotlib.pyplot as plt
                            import matplotlib.image as mpimg
                            img = mpimg.imread(
                                str(C.path / "visualize_classifiers" / id_baseline / (filename + ".png")))
                            imgplot = plt.imshow(img)
                            plt.show()

                    w = World(test_env)
                    w.draw_frame()
                    w.draw_lattice()
                    w.draw_cases()
                    for i, (s, a, s_, c) in enumerate(zip(S, A, S_, value)):
                        color = np.array([0., 0., 0.])
                        for i in range(len(exp.source_collection.envs.items())):
                            color += c[i] * np.array(colors[i])
                        w.draw_rectangle(
                            (s[0] - w_p / 2, s[1] - h_p / 2, s[0] + w_p / 2, s[1] + h_p / 2),
                            color=(*color, 1. / repeat_context))
                    for (s, a, r_, s_) in zip(*gt_trajs):
                        w.draw_trajectory([(s, None, None, s_, None, None)],
                                          (1, 1, 1, (1 / repeat_sources) * len(sources_trajs)),
                                          line_width=1)
                    filename = "target=" + target_env_name + "_" + metric
                    w.save(str(C.path / "visualize_classifiers" / id_baseline / filename))
                    if C.show_plots:
                        import matplotlib.pyplot as plt
                        import matplotlib.image as mpimg
                        img = mpimg.imread(str(C.path / "visualize_classifiers" / id_baseline / (filename + ".png")))
                        imgplot = plt.imshow(img)
                        plt.show()
