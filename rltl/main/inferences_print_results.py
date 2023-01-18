from collections import OrderedDict

from rltl.main.cvae import HYPER_VAE
from rltl.main.gan import HYPER_GAN, HYPER_NN
from rltl.utils.utils_os import list_checker, override_dictionary
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.utils_os import makedirs, save_object, load_object
import numpy as np


def compute_mus_and_sigmas(data):
    d = {}
    for sample in data:
        s = sample["s"]
        a = sample["a"]
        s_ = sample["s_"]
        if not (type(s) == np.ndarray):
            s = s.numpy()

        if not (type(a) == np.ndarray):
            a = a.numpy()

        if tuple(s) not in d:
            d[tuple(s) + tuple(a)] = []
        d[tuple(s) + tuple(a)].append(s_)
    mus_ground_truth = {}
    sigmas_ground_truth = {}
    for sa, all_s_ in d.items():
        mus_ground_truth[tuple(sa)] = np.mean(all_s_, axis=1)
        sigmas_ground_truth[tuple(sa)] = np.std(all_s_, axis=1)

    return mus_ground_truth, sigmas_ground_truth


def run(C, gan_baselines, envs_to_test):
    exp = ExperimentData(C.path, C).load(dont_load_models=True)  # .load()

    common_parameters = {**C["learn_gans"]}
    common_parameters_classifier = {**C["learn_classifiers"]}
    baselines_classifiers = common_parameters_classifier["baselines"]

    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]
    del common_parameters_classifier["baselines"]
    labels = []
    colors = []
    # for id_baseline, config_baseline in baselines.items():
    #     if gan_baselines is None or (gan_baselines is not None and id_baseline in gan_baselines):
    #         colors.append(tuple(config_baseline["color"]))
    results_reward = {}
    results_dynamics = {}

    for collection in [exp.source_collection, exp.target_collection]:
        for i_env, (env_name, (env_creator, config)) in enumerate(collection.envs.items()):
            print("==============================================")
            print("{} {}".format(env_name, config))
            print("==============================================")
            if list_checker(envs_to_test, env_name):
                labels.append(env_name)

                path_dyna = C.path / "inferences" / "samples" / "ground_truth" / "dynamics"
                data_dynamics_gt = load_object(str(path_dyna / (env_name + ".pickle")))
                if data_dynamics_gt is None:
                    raise Exception("Missing ground truth samples for {}+{}".format(id_baseline,env_name))
                mus_ground_truth, sigmas_ground_truth = compute_mus_and_sigmas(data_dynamics_gt)

                for id_baseline, config_baseline in baselines.items():
                    if list_checker(gan_baselines, id_baseline):
                        for res, xx in [(results_reward, "reward"), (results_dynamics, "dynamics")]:
                            # config_baseline = override_dictionary(common_parameters, config_baseline)
                            # if config_baseline["type"] in [HYPER_GAN, HYPER_NN, HYPER_VAE]:
                            #     context_type = C["inferences"]["context_type"]
                            # else:
                            #     context_type = "no_context"

                            # if context_type != "bypass" or (context_type == "bypass" and "source" in env_name):

                            for context_type in ["bypass","nothing","no_context"]:
                                path_dyna = C.path / "inferences" / "samples" / id_baseline / xx
                                if (path_dyna / (context_type+"_"+env_name + ".pickle")).exists():
                                    print("proceeding [{}] baseline={} (path={})".format(xx,id_baseline,path_dyna))
                                    data_dynamics = load_object(str(path_dyna / (context_type+"_"+env_name + ".pickle")))

                                    mus, sigmas = compute_mus_and_sigmas(data_dynamics)

                                    loss_mu = []
                                    loss_sigma = []
                                    loss_l1 = []
                                    for i in mus.keys():
                                        mu_ground, sigma_ground = mus_ground_truth[i], sigmas_ground_truth[i]
                                        mu_baseline, sigma_baseline = mus[i], sigmas[i]
                                        loss_mu.append(np.mean(np.abs(mu_ground - mu_baseline)))
                                        loss_sigma.append(np.mean(np.abs(sigma_ground - sigma_baseline)))

                                    if id_baseline not in res:
                                        color = list(config_baseline["color"])
                                        if context_type == "bypass":
                                            color[3] = 0.5
                                        res[(id_baseline,context_type)] = {
                                            "sigma": OrderedDict(),
                                            "mu": OrderedDict(),
                                            "color": tuple(color)}
                                    res[(id_baseline,context_type)]["mu"][env_name] = np.mean(loss_mu)
                                    res[(id_baseline,context_type)]["sigma"][env_name] = np.mean(loss_sigma)
                                else:
                                    print("Can't proceed [{}] baseline={}, pickle path does not exist ({}, {})".format(xx,
                                                                                                               id_baseline,
                                                                                                               path_dyna,
                                                                                                               context_type))
    if len(results_dynamics) > 0:
        plot_results(C, "dynamics (sigma)", labels, results_dynamics, "sigma")
        plot_results(C, "dynamics (mu)", labels, results_dynamics, "mu")
        save_object(results_dynamics, C.path / "inferences" / "results_dynamics.pickle")
        # create_latex_tab(results_dynamics)
        # plot_results(C, "dynamics (l1)", labels, results_dynamics, "l1")
    if len(results_reward) > 0:
        plot_results(C, "reward (sigma)", labels, results_reward, "sigma")
        plot_results(C, "reward (mu)", labels, results_reward, "mu")
        save_object(results_dynamics, C.path / "inferences" / "results_reward.pickle")
        # plot_results(C, "reward (l1)", labels, results_reward, "l1")


def plot_results(C, title, labels, results, sigma_or_mu):
    # print(title, results)
    path = C.path / "inferences"
    makedirs(path)
    import matplotlib.pyplot as plt
    import numpy as np

    rect_dynamics = {}

    x = np.arange(len(labels))  # the label locations

    width = 0.5  # the width of the bars
    width_single_bar = width / len(results)
    fig, ax = plt.subplots()

    offset = width / len(results)

    for i, (baseline, result) in enumerate(results.items()):
        # print(baseline, result)
        values = list(result[sigma_or_mu].values())
        # print(values)
        rect_dynamics[baseline] = ax.bar(x - width / 2 + offset * i, values,
                                         width_single_bar,
                                         label=baseline, color=result["color"])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('l1 (wrt ground truth)')
    plt.title(title)
    plt.xticks(x, labels, rotation='vertical')
    plt.legend()
    fig.tight_layout()
    plt.savefig(str(path / (title + ".png")))
    if C.show_plots:
        plt.show()
