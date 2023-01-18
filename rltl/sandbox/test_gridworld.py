from rltl.envs.gridworld.factory import generate_random_model, generate_test_0, grid0
import numpy as np

import time

# e,_ = generate_random_model(std=(0.5, 0.5),nb_path_to_goal=10,A=[(0., 1.), (1., 0.)])
e = grid0(wind=lambda x, y: (0, -0.4))
e = e()
len_trajs = []
obs = []
for i in range(1):
    done = False
    k = 0
    e.reset()
    while not done:
        o, r, done, info = e.step(e.action_space.sample())
        e.render(save_prefix="episode_{}".format(i))
        obs.append(o)
        k += 1
        if done:
            time.sleep(1)
    len_trajs.append(k)
print("episode len", np.mean(len_trajs, axis=0))
print("mean_o", np.mean(obs, axis=0))
