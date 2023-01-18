from rltl.envs.observation_to_numpy import obs_to_np_factory
from rltl.main.classifier import BAYESIAN_CLASSIFIER
from rltl.main.cvae import HYPER_VAE, SamplerCVAE
from rltl.main.gan import CLASSIC, SUPER_GAN, AGGLOMERATED_GAN, HYPER_GAN, HYPER_NN
from rltl.utils.utils_os import list_checker, override_dictionary
from rltl.main.policy import RandomPolicy, SingleActionPolicy
from rltl.main.print_cc import create_heatmap
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.registry import R, EnvGridWorld
from rltl.utils.replay_memory import Memory
from rltl.utils.transition import TransitionGym
from rltl.utils.utils_os import makedirs, save_object
from rltl.utils.utils_rl import exploration, change_env_state_gridworld, rollout, reverse_one_hot
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def run(C, gan_baselines, envs_to_test, path_to_save=None, file_prefix=""):
    exp = ExperimentData(C.path, C).load(dont_load_models=True)

    common_parameters = {**C["learn_gans"]}
    common_parameters_classifier = {**C["learn_classifiers"]}
    baselines_classifiers = common_parameters_classifier["baselines"]

    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]
    del common_parameters_classifier["baselines"]
    labels = []
    colors = []
    for id_baseline, config_baseline in baselines.items():
        if list_checker(gan_baselines, id_baseline):
            colors.append(tuple(config_baseline["color"]))
    results_reward = {}
    results_dynamics = {}

    path_to_save = C.path if path_to_save is None else path_to_save

    makedirs(str(path_to_save / "inferences" / "worlds"))
    obs_to_np = obs_to_np_factory(C["obs_to_np"])
    for collection in [exp.source_collection, exp.target_collection]:
        for i_env, (env_name, (env_creator, config)) in enumerate(collection.envs.items()):
            if list_checker(envs_to_test, env_name):
                labels.append(env_name)
                m = exploration(
                    env_creator=env_creator,
                    change_env_init_state=change_env_state_gridworld,
                    one_hot_action=C["onehot_action"],
                    **C["inferences"]["exploration"])

                source_samples = m.memory
                S, A, R_, S_ = ([obs_to_np(x.s) for x in source_samples],
                                [x.a for x in source_samples],
                                [x.r_ for x in source_samples],
                                [obs_to_np(x.s_) for x in source_samples])
                steps = tf.data.Dataset.from_tensor_slices((S, A, R_, S_))
                steps = steps.batch(len(S))

                for (s, a, r_, s_) in steps:
                    a = tf.cast(a, tf.float32)
                    s = tf.cast(s, tf.float32)
                    s_ = tf.cast(s_, tf.float32)
                    r_ = tf.cast(r_, tf.float32)

                from rltl.envs.gridworld.world import World
                test_env = env_creator()
                world = None
                if isinstance(test_env, EnvGridWorld):

                    if len(baselines.items()) > 1:
                        world = World(test_env)
                        world.draw_frame()
                        world.draw_cases()
                        world.draw_lattice()

                    if "plot_ground_truth" in C["inferences"] and C["inferences"]["plot_ground_truth"]:
                        data = ground_truth_inferences(env_name, s, a, C, env_creator, (1, 1, 1, 0.1), world,
                                                       path_to_save,
                                                       file_prefix)
                        path_dyna = path_to_save / "inferences" / "samples" / "ground_truth" / "dynamics"
                        makedirs(path_dyna)
                        if data:
                            save_object(data, str(path_dyna / (file_prefix + env_name + ".pickle")))

                actual_nb_of_baseline = 0
                for id_baseline, config_baseline in baselines.items():
                    if list_checker(gan_baselines, id_baseline):
                        if id_baseline in exp.learn_gans_data.baselines:
                            exp.learn_gans_data.load_baseline(id_baseline)
                            baseline = exp.learn_gans_data.baselines[id_baseline]
                            # config_baseline = {**config_baseline, **common_parameters}
                            config_baseline = override_dictionary(common_parameters, config_baseline)
                            if "classifier_baseline" in config_baseline:
                                config_classifier = override_dictionary(common_parameters_classifier,
                                                                        baselines_classifiers[
                                                                            config_baseline["classifier_baseline"]])
                            else:
                                config_classifier = None
                            if config_baseline["type"] in [HYPER_GAN, HYPER_NN, HYPER_VAE]:
                                context_type = C["inferences"]["context_type"]
                            else:
                                context_type = "no_context"
                            if context_type != "bypass" or (context_type == "bypass" and "source" in env_name):
                                data_s_ = inferences(
                                    env_name, id_baseline, s, a, r_, s_, C, exp,
                                    config_baseline, config_classifier, baseline, env_creator,
                                    "dynamics",
                                    tuple(config_baseline["color"]), world, path_to_save, file_prefix,
                                    context_type,
                                    obs_to_np)
                                path_dyna = C.path / "inferences" / "samples" / id_baseline / "dynamics"
                                makedirs(path_dyna)
                                if data_s_:
                                    save_object(data_s_,
                                                str(path_dyna / (
                                                        file_prefix + context_type + "_" + env_name + ".pickle")))
                                actual_nb_of_baseline += 1
                        else:
                            print("warning: baseline {} does not exist in the data".format(id_baseline))
                if actual_nb_of_baseline > 1:
                    world.save(
                        str(path_to_save / "inferences" / "worlds" / (file_prefix + context_type + "_" + env_name)))
                if actual_nb_of_baseline == 0:
                    raise Exception("Nothing has been proceed ({} ... {})".format(gan_baselines, baselines.keys()))
                # if C.show_plots and actual_nb_of_baseline > 1:
                #     import matplotlib.pyplot as plt
                #     import matplotlib.image as mpimg
                #     img = mpimg.imread(str(path_to_save / "inferences" / "worlds" / (
                #             file_prefix + "_" + env_name)) + ".png")
                #     imgplot = plt.imshow(img)
                # #     plt.show()


def ground_truth_inferences(env_name, s, a, C, env_creator, fake_color, world, path_to_save, file_prefix):
    res = []
    from rltl.envs.gridworld.world import World
    test_env = env_creator()

    w = World(test_env)
    w.draw_frame()
    w.draw_cases()
    w.draw_lattice()
    for i in range(len(s)):
        true_s = s[i]
        data = {"s": true_s, "a": a[i], "s_": []}
        for _ in range(C["inferences"]["n_repeat_forward"]):
            samples, _ = rollout(
                env_creator,
                SingleActionPolicy(int(reverse_one_hot(a[i]))),
                k=1,
                init_sample=TransitionGym(s[i], None, None, None, None, None),
                onehot_action=C["onehot_action"],
                change_env_init_state=change_env_state_gridworld)
            s_ = samples.memory[0].s_

            # data = {"s": true_s, "a": a[i], "s_": s_}

            data["s_"].append(s_)

            true_s = np.array(s[i])
            true_s_ = np.array(s_)

            if test_env.normalise_state:
                true_s[0] = true_s[0] * test_env.w
                true_s[1] = true_s[1] * test_env.h
                true_s_[0] = true_s_[0] * test_env.w
                true_s_[1] = true_s_[1] * test_env.h

            w.draw_trajectory([(true_s, None, None, true_s_, None, None)], fake_color, line_width=1)
            if world is not None:
                world.draw_trajectory([(true_s, None, None, true_s_, None, None)], fake_color, line_width=1)
        res.append(data)

    w.save(str(path_to_save / "inferences" / "worlds" / (file_prefix + env_name + "_" + "ground_truth")))
    if C.show_plots:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(
            str(path_to_save / "inferences" / "worlds" / (file_prefix + env_name + "_" + "ground_truth")) + ".png")
        imgplot = plt.imshow(img)
        plt.show()

    return res


def inferences(env_name, id_baseline, s, a, r_, s_, C, exp, config_baseline, config_classifier, baseline, env_creator,
               type_model,
               fake_color, world, path_to_save, file_prefix, context_type,obs_to_np):
    if type_model == "reward":
        x_ = tf.reshape(r_, (-1, 1))
    else:
        x_ = s_
    if config_baseline["type"] in [HYPER_GAN, HYPER_NN, HYPER_VAE]:
        classifier_baseline = config_baseline["classifier_baseline"]
        type_classifier = C["learn_classifiers"]["baselines"][classifier_baseline]["type"]
        is_bayesian = type_classifier == BAYESIAN_CLASSIFIER
        exp.learn_classifiers_data.load_baseline(classifier_baseline)
        baseline_classifier = exp.learn_classifiers_data.baselines[classifier_baseline]
        #
        if type_model == "reward":
            context_model = baseline_classifier.get_model("best", type_classifier, "reward", "classifier", "all_envs")
        else:
            context_model = baseline_classifier.get_model("best", type_classifier, "dynamics", "classifier", "all_envs")
        if context_model is not None:
            # repeat inference and get mu and sigma
            if is_bayesian:
                contexts = []
                for _ in range(C["inferences"]["repeat_bayesian_inferences"]):
                    contexts.append(context_model([s, a, x_],
                                                  training="training" in config_classifier and config_classifier[
                                                      "training"]))
                mu_c = np.mean(contexts, axis=0)
                sigma_c = np.std(contexts, axis=0)

            else:
                mu_c = context_model([s, a, x_],
                                     training="training" in config_classifier and config_classifier["training"]).numpy()
                sigma_c = np.zeros(mu_c.shape)
        else:
            raise Exception("No context model")
    if config_baseline["type"] == CLASSIC:
        name_env = env_name
    else:
        name_env = "all_envs"
    generative_type = config_baseline["generative_type"]
    if generative_type == "GAN":
        model = baseline.get_model("best", config_baseline["type"], type_model, "G", name_env)
    elif generative_type == "CVAE":
        decoder = baseline.get_model("best", config_baseline["type"], type_model, "decoder", name_env)
        if decoder is None:
            raise Exception("Can't find decoder for {}, name_env={}".format(id_baseline, name_env))
        model = SamplerCVAE(decoder=decoder, config_cvae=config_baseline)
    else:
        raise Exception("unknown generative type: {}".format(generative_type))
        # D = baseline.get_model("best", config_baseline["type"], type_model, "D", name_env)

    if model is not None:

        z_size = config_baseline["z_size"]
        fake_xxx_ = []

        # if config_baseline["type"] in [HYPER_GAN, HYPER_NN, HYPER_VAE]:
        #     context_type = C["inferences"]["context_type"]
        # else:
        #     context_type = "no_context"
        proceed = True
        for _ in range(C["inferences"]["n_repeat_forward"]):
            if generative_type == "GAN":
                z = tf.cast(np.random.normal(-1.0, 1.0, size=[len(s), z_size]), tf.float32)
            # elif generative_type == "CVAE":
            #     z = tf.cast(tf.random.normal(shape=(len(s), z_size)), tf.float32)

            S, A, Z, CTX = [], [], [], []
            for i in range(len(s)):
                S.append(s[i])
                A.append(a[i])
                if generative_type == "GAN":
                    Z.append(z[i])
                if config_baseline["type"] in [HYPER_GAN, HYPER_NN, HYPER_VAE]:
                    if isinstance(C["inferences"]["sigma_eps"], str):
                        sigma_eps = exp.threshold_sigma[classifier_baseline][C["inferences"]["sigma_eps"]]
                    else:
                        sigma_eps = float(C["inferences"]["sigma_eps"])

                    if is_bayesian and np.mean(sigma_c[i]) > sigma_eps:
                        # use the context vector as it is
                        c_i = mu_c[i]
                    else:

                        # print("context_type='{}'".format(context_type))
                        if context_type == "softmax":
                            distrib = np.exp(mu_c[i]) / sum(np.exp(mu_c[i]))
                        elif context_type == "bypass":
                            distrib = np.zeros(len(exp.envs_source))
                            distrib[int(env_name[-1])] = 1
                            if "target" in env_name:
                                print(
                                    "[WARNING] Cannot bypass target env, because context vector describes source only")
                                return None
                        elif context_type == "max":
                            i = np.argmax(mu_c[i])
                            distrib = np.zeros(len(exp.envs_source))
                            distrib[i] = 1
                        elif context_type == "nothing":
                            distrib = mu_c[i] / np.sum(mu_c[i])  # in case it doesn't sum EXACTLY to one
                        else:
                            raise Exception("unknown type for context: {}".format(context_type))
                        idx = np.random.choice(range(len(distrib)), p=distrib)
                        cbis = np.zeros(len(exp.envs_source))
                        cbis[idx] = 1
                        c_i = cbis

                    CTX.append(c_i)
            type_baseline = config_baseline["type"]
            if type_baseline == HYPER_GAN:
                G_input = tf.data.Dataset.from_tensor_slices((S, A, Z, CTX))
            elif type_baseline == HYPER_VAE:
                G_input = tf.data.Dataset.from_tensor_slices((S, A, CTX))
            elif type_baseline == HYPER_NN:
                G_input = tf.data.Dataset.from_tensor_slices((S, A, CTX))
            elif type_baseline in [CLASSIC, AGGLOMERATED_GAN]:
                if generative_type == "CVAE":
                    G_input = tf.data.Dataset.from_tensor_slices((S, A))
                elif generative_type == "GAN":
                    G_input = tf.data.Dataset.from_tensor_slices((S, A, Z))
            else:
                raise Exception("unkown config: {}".format(type_baseline))
            G_input = G_input.batch(len(s))
            # G_input = tf.convert_to_tensor(G_input)
            for ix, x in enumerate(G_input):
                if ix > 0:
                    raise Exception()
                if generative_type == "GAN":
                    one_forward_pass = model(x, training="training" in config_baseline and config_baseline["training"])
                elif generative_type == "CVAE":
                    one_forward_pass = model.sample(*x)
            fake_xxx_.append(one_forward_pass)

        res = []
        for i in range(len(s)):
            data = {"s": s[i].numpy(), "a": a[i].numpy(), "s_": []}
            for one_forward_pass in fake_xxx_:
                data["s_"].append(one_forward_pass[i].numpy())
            res.append(data)

        test_env = env_creator()
        if isinstance(test_env, EnvGridWorld):
            from rltl.envs.gridworld.world import World

            w = World(test_env)
            w.draw_frame()
            w.draw_cases()
            w.draw_lattice()
            for data in res:
                true_s = np.array(data["s"])

                if test_env.normalise_state:
                    true_s[0] = true_s[0] * test_env.w
                    true_s[1] = true_s[1] * test_env.h

                fake_color = list(fake_color)
                fake_color[3] = C["inferences"]["fake_color_alpha"]
                for s_ in data["s_"]:
                    fake_s_ = np.array(s_)
                    if test_env.normalise_state:
                        fake_s_[0] = fake_s_[0] * test_env.w
                        fake_s_[1] = fake_s_[1] * test_env.h

                    w.draw_trajectory([(true_s, None, None, fake_s_, None, None)], fake_color, line_width=1)
                    if world is not None:
                        world.draw_trajectory([(true_s, None, None, fake_s_, None, None)], fake_color, line_width=1)

            w.save(
                str(path_to_save / "inferences" / "worlds" / (
                        file_prefix + context_type + "_" + env_name + "_" + id_baseline)))
        if isinstance(test_env, EnvGridWorld):
            if C.show_plots:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                plt.title(env_name + "_" + id_baseline)
                img = mpimg.imread(
                    str(path_to_save / "inferences" / "worlds" / (
                            file_prefix + context_type + "_" + env_name + "_" + id_baseline)) + ".png")
                imgplot = plt.imshow(img)
                plt.show()

        return res
    return None
