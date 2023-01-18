from rltl.main.gan import CLASSIC, SUPER_GAN, AGGLOMERATED_GAN, HYPER_GAN
from rltl.main.policy import RandomPolicy
from rltl.main.print_cc import create_heatmap
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.registry import R
from rltl.utils.utils_os import makedirs, override_dictionary
from rltl.utils.utils_rl import exploration, change_env_state_gridworld
import numpy as np
import tensorflow as tf


def run(C, gan_baselines):
    exp = ExperimentData(C.path, C).load()

    common_parameters = {**C["learn_gans"]}
    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]

    labels = []
    results_reward = {}
    results_dynamics = {}

    for collection in [exp.source_collection, exp.target_collection]:
        for i_env, (env_name, (env_creator, config)) in enumerate(collection.envs.items()):
            labels.append(env_name)
            print("==============================================")
            print("==============================================")
            print("{} {}".format(env_name, config))
            print("==============================================")
            print("==============================================")

            m = exploration(
                env_creator=env_creator,
                change_env_init_state=change_env_state_gridworld,
                one_hot_action=C["onehot_action"],
                **C["test_models"]["exploration"])

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

            from rltl.envs.gridworld.world import World
            test_env = env_creator()
            world = World(test_env)
            world.draw_frame()
            world.draw_cases()
            world.draw_lattice()

            for id_baseline, config_baseline in baselines.items():
                if gan_baselines is None or (gan_baselines is not None and id_baseline in gan_baselines):
                    if id_baseline not in results_reward:
                        results_reward[id_baseline] = []
                        results_dynamics[id_baseline] = []
                    print("[baseline={}]".format(id_baseline))
                    baseline = exp.learn_gans_data.baselines[id_baseline]
                    config_baseline = override_dictionary(common_parameters, config_baseline)
                    # config_baseline = {**config_baseline, **common_parameters}

                    l1_s_ = test_model(env_name, id_baseline, s, a, r_, s_, C, exp,
                                       config_baseline, baseline, env_creator, "dynamics",
                                       world, tuple(config_baseline["color"]))
                    l1_r_ = test_model(env_name, id_baseline, s, a, r_, s_, C, exp,
                                       config_baseline, baseline, env_creator, "reward",
                                       world, tuple(config_baseline["color"]))
                    results_dynamics[id_baseline].append(np.mean(l1_s_))
                    results_reward[id_baseline].append(np.mean(l1_r_))

            world.save(str(C.path / "test_models" / (env_name)))
            if C.show_plots:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                img = mpimg.imread(str(C.path / "test_models" / (env_name)) + ".png")
                imgplot = plt.imshow(img)
                plt.show()

    plot_results(C, "dynamics", labels, results_dynamics)
    plot_results(C, "reward", labels, results_reward)


def plot_results(C, title, labels, results_dynamics):
    print(title, results_dynamics)
    path = C.path / "test_models"
    makedirs(path)
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    rect_dynamics = {}

    x = np.arange(len(labels))  # the label locations

    width = 0.5  # the width of the bars

    fig, ax = plt.subplots()

    offset = width / len(results_dynamics)

    for i, (baseline, result) in enumerate(results_dynamics.items()):
        rect_dynamics[baseline] = ax.bar(x - width / 2 + offset * i, result, width / len(results_dynamics),
                                         label=baseline)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('l1')
    plt.title(title)
    plt.xticks(x, labels, rotation='vertical')
    plt.legend()
    fig.tight_layout()
    plt.savefig(str(path / (title + ".png")))
    if C.show_plots:
        plt.show()


def test_model(env_name, id_baseline, s, a, r_, s_, C, exp, config_baseline, baseline, env_creator, type_model, world,
               fake_color):
    if config_baseline["type"] == HYPER_GAN:
        baseline_classifier = exp.learn_classifiers_data.baselines[config_baseline["classifier_baseline"]]
        model_dynamics = baseline_classifier.get_model("best", "classifier", "dynamics", "classifier", "all_envs")
        c = model_dynamics([s, a, s_])

    if config_baseline["type"] == CLASSIC:
        name_env = env_name
    else:
        name_env = "all_envs"

    G = baseline.get_model("best", config_baseline["type"], type_model, "G", name_env)
    D = baseline.get_model("best", config_baseline["type"], type_model, "D", name_env)

    reconstruction_errors = []

    if G is not None and D is not None:

        if type_model == "reward":
            x_ = tf.reshape(r_, (-1, 1))
        else:
            x_ = s_

        input = [s, a]

        z_size = config_baseline["z_size"]
        if z_size > 0:
            z = tf.cast(np.random.normal(-1.0, 1.0, size=[len(s), z_size]), tf.float32)
            input.append(z)

        if config_baseline["type"] == SUPER_GAN:
            acc_dynamic = D([s, a, x_])
            if config_baseline["factorize_fake_output"]:
                acc_dynamic, _ = acc_dynamic

            if C["test_models"]["average_accuracy"]:
                acc_dynamic = np.mean(acc_dynamic, axis=0)
                print("acc_dynamic", acc_dynamic)
                tilled_acc_dynamic = tf.tile(tf.expand_dims(acc_dynamic, axis=0), (len(s), 1))
                input.append(tilled_acc_dynamic)
            else:
                input.append(acc_dynamic)
        if config_baseline["type"] == HYPER_GAN:
            input.append(c)

        fake_xxx_ = []
        for _ in range(C["test_models"]["n_repeat_forward"]):
            input = [s, a]
            if z_size > 0:
                z = tf.cast(np.random.normal(-1.0, 1.0, size=[len(s), z_size]), tf.float32)
                input.append(z)
            if config_baseline["type"] == SUPER_GAN:
                if C["test_models"]["average_accuracy"]:
                    input.append(tilled_acc_dynamic)
                else:
                    input.append(acc_dynamic)
            if config_baseline["type"] == HYPER_GAN:
                input.append(c)
            fake_xxx_.append(G(input))

        from rltl.envs.gridworld.world import World
        test_env = env_creator()
        w = World(test_env)
        w.draw_frame()
        w.draw_cases()
        w.draw_lattice()
        for i in range(len(s)):
            true_s = np.array(s[i])
            true_s_ = np.array(s_[i])

            if test_env.normalise_state:
                true_s[0] = true_s[0] * test_env.w
                true_s[1] = true_s[1] * test_env.h
                true_s_[0] = true_s_[0] * test_env.w
                true_s_[1] = true_s_[1] * test_env.h

            w.draw_trajectory([(true_s, None, None, true_s_, None, None)], (255, 255, 255, 1), line_width=1)
            world.draw_trajectory([(true_s, None, None, true_s_, None, None)], (255, 255, 255, 1), line_width=1)

            for fake_x_ in fake_xxx_:
                fake_s_ = np.array(fake_x_[i])
                if test_env.normalise_state:
                    fake_s_[0] = fake_s_[0] * test_env.w
                    fake_s_[1] = fake_s_[1] * test_env.h
                # fake_s_ = np.array(fake_x_[i])
                w.draw_trajectory([(true_s, None, None, fake_s_, None, None)], fake_color, line_width=1)
                world.draw_trajectory([(true_s, None, None, fake_s_, None, None)], fake_color, line_width=1)

        makedirs(str(C.path / "test_models"))
        w.save(str(C.path / "test_models" / (env_name + "_" + id_baseline)))
        if C.show_plots:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            img = mpimg.imread(str(C.path / "test_models" / (env_name + "_" + id_baseline)) + ".png")
            imgplot = plt.imshow(img)
            plt.show()

        reconstruction_error = np.abs(x_ - np.mean(fake_xxx_, axis=0))
        for i in range(5):
            print("s_={}, fake_s_={}".format(x_[i], fake_x_[i]))
        reconstruction_errors.append(np.mean(reconstruction_error))

    return reconstruction_errors
