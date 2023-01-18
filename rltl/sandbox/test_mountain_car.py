from gym.envs.classic_control import MountainCarEnv
import gym
from rltl.envs.cartpole import ParametricCartPoleEnv
from rltl.envs.mountain_car import ParametricMountainCarEnv
import numpy as np
# e = ParametricMountainCarEnv(gravity=ParametricMountainCarEnv.DEFAULT_GRAVITY * 0.001)
# e = ParametricMountainCarEnv(force=ParametricMountainCarEnv.DEFAULT_FORCE *10.)
from rltl.utils.registry import R

# e = ParametricMountainCarEnv(force=ParametricMountainCarEnv.DEFAULT_FORCE *100)
# e = ParametricCartPoleEnv(length=ParametricMountainCarEnv.DEFAULT_FORCE *100)

collection = "cp_length_easy"
# collection = "debug_high_length"

d_source, d_test = R.get_envs_collection(collection)()
n_episodes = 1000

for isource, (name, (env_creator,config)) in enumerate(d_source.envs.items()):
    print("name={} config={}".format(name, config))
    e = env_creator()
    len_trajs = []
    obs = []
    for i in range(n_episodes):
        done = False
        k = 0
        e.reset()
        while not done:
            o, r, done, info = e.step(e.action_space.sample())
            # e.render()
            obs.append(o)
            k += 1
        len_trajs.append(k)
    print("episode len", np.mean(len_trajs, axis=0))
    print("mean_o", np.mean(obs, axis=0))
