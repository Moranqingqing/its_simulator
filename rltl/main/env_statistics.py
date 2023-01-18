import gym

from rltl.main.policy import RandomPolicy
from rltl.utils.experiment_state import ExperimentData, EnvData
import logging

from rltl.utils.transition import TransitionGym
from rltl.utils.utils_rl import rollouts, rollout, change_env_state_gridworld, exploration
import pandas as pd
import numpy as np
from rltl.envs.gridworld.world import World
from rltl.utils.utils_os import makedirs
from rltl.utils.replay_memory import Memory


def run(C):
    exp = ExperimentData(C.path, C)

    logging.basicConfig(level=logging.INFO)

    for type_coll, env_coll in [("source", exp.source_collection), ("target", exp.target_collection)]:
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("generating {} samples from collection {}".format(type_coll, env_coll))
        envs = env_coll.envs.items()
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        makedirs(C.path / "env_statistics")

        for name, (env_creator, env_config) in envs:
            test_env = env_creator()
            print("GENERATING SAMPLES FOR ENV: {}, config={}".format(name, env_config))
            m = exploration(
                env_creator=env_creator,
                change_env_init_state=change_env_state_gridworld,
                one_hot_action=C["onehot_action"],
                **C["env_statistics"]["exploration"])
            all_s = []
            all_r_ = []

            for s, a, r_, s_, _, _ in m.sample(5):
                print("s={} a={} -> s'={}".format(s, a, s_))

            trajectories = []
            trajectory = []

            for sample in m.memory:
                s, a, r_, s_, done, info = sample
                all_s.append(sample[0])
                all_r_.append(sample.r_)
                true_s = np.array(s)
                true_s_ = np.array(s_)
                if test_env.normalise_state:
                    true_s[0] = true_s[0] * test_env.w
                    true_s[1] = true_s[1] * test_env.h
                    true_s_[0] = true_s_[0] * test_env.w
                    true_s_[1] = true_s_[1] * test_env.h

                trajectory.append((true_s, a, r_, true_s_, done, info))
                if C["env_statistics"]["exploration"]["type"] == "random":
                    if done:
                        trajectories.append(trajectory)
                        trajectory = []
                elif C["env_statistics"]["exploration"]["type"] == "uniform":
                    trajectories.append(trajectory)
                    trajectory = []
                elif C["env_statistics"]["exploration"]["type"] == "grid":
                    trajectories.append(trajectory)
                    trajectory = []
            w = World(e=test_env)
            w.draw_frame()
            w.draw_cases()
            w.draw_lattice()
            w.draw_trajectories(trajectories,rgba=(1,1,1,C["env_statistics"]["alpha"]),line_width=1)

            w.save(str(C.path / "env_statistics" / ("traj_" + name)))
            if C.show_plots:
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                img = mpimg.imread(str(C.path / "env_statistics" / ("traj_" + name))+".png")
                imgplot = plt.imshow(img)
                plt.show()
            all_s = np.stack(all_s)
            all_r_ = np.stack(all_r_)
            df = pd.DataFrame.from_records(
                m.memory,
                columns=TransitionGym._fields
            )
            print("mean", np.mean(all_s, axis=0))
            print("std", np.std(all_s, axis=0))
            print("mean r_", np.std(all_r_, axis=0))
            # print(df.mean())
