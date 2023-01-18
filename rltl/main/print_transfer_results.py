from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
#import pyvirtualdisplay
from rltl.utils.utils_os import list_checker, override_dictionary
from rltl.utils.experiment_state import ExperimentData
from rltl.utils.utils_os import makedirs

num_cores = 4

DQN = "dqn"
DQN_EXTRA_UPDATES = "dqn_extra_updates"
DQN_UPPER_BOUND = "dqn_upper_bound"
SOLUTION_1 = "solution_1"
SOLUTION_3 = "solution_3"
SOLUTION_2 = "solution_2"
DYNA_MODEL_LEARNING = "dyna_model_learning"
DYNA_AGGLO = "dyna_agglo"
DYNA_RDM = "dyna_rdm"
DYNA_PRETRAINED = "dyna_pretrained"
DYNA_PERFECT = "dyna_perfect"


def run(C,transfer_baselines):
    exp = ExperimentData(C.path,C).load(dont_load_models=True)
    common_parameters = {**C["transfer"]}
    baselines = common_parameters["baselines"]
    del common_parameters["baselines"]
    # such that we start the result for all the baselines for the same env
    makedirs(C.path / "transfer_results")
    for ic, collection in enumerate([exp.source_collection, exp.target_collection]):
        for env_name, (env_creator,config) in collection.envs.items():
            fig, ax = plt.subplots(1)
            plot_something = False
            for id_baseline, config_baseline in baselines.items():
                if list_checker(transfer_baselines,id_baseline):
                # if transfer_baselines is None or (transfer_baselines is not None and id_baseline in transfer_baselines):
                    egreedys = []
                    greedys = []
                    for i_run in range(C["transfer"]["nb_runs"]):
                        path = C.path / "transfer" / id_baseline / env_name / "run_{}".format(i_run)
                        greedy_path = path / "greedy.npy"
                        if greedy_path.exists():
                            plot_something = plot_something or True
                            print("[RUN={}][collection={}][env_name={} / config={}][baseline={}]"
                                  .format(i_run, ["source", "target"][ic], env_name, config, id_baseline))
                            # config_baseline = {**config_baseline, **common_parameters}
                            config_baseline = override_dictionary(common_parameters, config_baseline)
                            greedy = np.load(path / "greedy.npy")
                            egreedy = np.load(path / "egreedy.npy")
                            egreedys.append(egreedy)
                            greedys.append(greedy)
                    if len(greedys) > 0:
                        greedys = np.stack(greedys,axis=2)
                        x = np.mean(greedys, axis=2)[0]
                        y = np.mean(greedys, axis=2)[1]
                        std_y = np.std(greedys, axis=2)[1]
                        plt.plot(x, y, label=id_baseline)
                        ax.fill_between(x, y + std_y, y - std_y, facecolor='blue', alpha=0.5)
            if plot_something:
                plt.title(env_name+" "+str(config))
                plt.legend()
                plt.savefig( str(C.path / "transfer_results" / (env_name +".png")))
                if C.show_plots:
                    plt.show()
