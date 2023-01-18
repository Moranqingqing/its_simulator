import gym

from rltl.envs.gridworld.envgridworld import EnvGridWorld
from rltl.envs.gridworld.world import World
from rltl.envs.observation_to_numpy import obs_to_np_factory
from rltl.main.policy import RandomPolicy
from rltl.utils.experiment_state import ExperimentData, EnvData
import logging
from rltl.utils.transition import TransitionGym

from rltl.utils.utils_rl import rollouts,rollout,exploration,change_env_state_gridworld

from rltl.utils.utils_os import makedirs, list_checker
from rltl.utils.replay_memory import Memory
import numpy as np
def run(C,envs_to_generate):
    exp = ExperimentData(C.path, C)

    obs_to_np = obs_to_np_factory(C["obs_to_np"])
    logging.basicConfig(level=logging.INFO)
    for type_coll, env_coll in [("source", exp.source_collection), ("target", exp.target_collection)]:
        print("generating {} samples from collection {}".format(type_coll, env_coll))
        envs = env_coll.envs.items()

        for name, (env_creator, env_config) in envs:
            if list_checker(envs_to_generate,name):
                print("GENERATING SAMPLES FOR ENV: {}, config={}".format(name, env_config))
                m = exploration(
                    env_creator=env_creator,
                    change_env_init_state=change_env_state_gridworld,
                    one_hot_action=C["onehot_action"],
                    **C["create_datasets"]["exploration"])
                for sample in m.sample(10):
                    print(sample)
                ees = EnvData()
                ees.set_name(name)
                ees.set_env_config(env_config)
                ees.set_memory(m)
                if type_coll == "source":
                    exp.add_exp_env_state_source(ees)
                elif type_coll == "target":
                    exp.add_exp_env_state_target(ees)
                else:
                    raise Exception()

                if C["create_datasets"]["save_source_trajectories"]:
                    all_s = []
                    all_s_ = []
                    all_r_ = []
                    test_env = env_creator()
                    for s, a, r_, s_, _, _ in m.sample(5):
                        print("s={} a={} -> s'={}".format(s, a, s_))

                    trajectories = []
                    trajectory = []
                    for sample in m.memory:
                        s, a, r_, s_, done, info = sample
                        s= obs_to_np(s)
                        s_ = obs_to_np(s_)
                        all_s.append(s)
                        all_s_.append(s_)
                        all_r_.append(r_)
                        true_s = np.array(s)
                        true_s_ = np.array(s_)

                        if isinstance(test_env, EnvGridWorld):
                            if  test_env.normalise_state:
                                true_s[0] = true_s[0] * test_env.w
                                true_s[1] = true_s[1] * test_env.h
                                true_s_[0] = true_s_[0] * test_env.w
                                true_s_[1] = true_s_[1] * test_env.h

                        trajectory.append((true_s, a, r_, true_s_, done, info))
                        if C["create_datasets"]["exploration"]["type"] == "random":
                            if done:
                                trajectories.append(trajectory)
                                trajectory = []
                        elif C["create_datasets"]["exploration"]["type"] == "uniform":
                            trajectories.append(trajectory)
                            trajectory = []
                        elif C["create_datasets"]["exploration"]["type"] == "grid":
                            trajectories.append(trajectory)
                            trajectory = []

                    if isinstance(test_env,EnvGridWorld):
                        w = World(e=test_env)
                        w.draw_frame()
                        w.draw_cases()
                        w.draw_lattice()
                        w.draw_source_trajectories(trajectories,alpha=C["create_datasets"]["alpha"])
                        makedirs(str(C.path / "create_datasets"))
                        w.save(str(C.path / "create_datasets" / ("traj_" + name)))
                        if C.show_plots:
                            import matplotlib.pyplot as plt
                            import matplotlib.image as mpimg
                            img = mpimg.imread(str(C.path / "create_datasets" / ("traj_" + name))+".png")
                            imgplot = plt.imshow(img)
                            plt.show()

                    # TODO, flatten dictionnary obs
                    all_s = np.stack(all_s)
                    all_s_ = np.stack(all_s_)
                    all_r_ = np.stack(all_r_)
                    print("mean", np.mean(all_s, axis=0))
                    print("std", np.std(all_s, axis=0))
                    print("mean r_", np.std(all_r_, axis=0))
                    print("mean abs s - s_", np.mean(abs(all_s - all_s_), axis=0))
    exp.save()
