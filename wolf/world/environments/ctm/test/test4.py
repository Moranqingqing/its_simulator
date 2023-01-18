import unittest
import os
from time import sleep

from wolf.world.environments.ctm.ctm_env import CtmEnv, CHANGE, EXTEND, RED, GREEN
import numpy as np


class Test(unittest.TestCase):

    def test0(self):
        np.random.seed(10)
        render_config = {
            "width": 900,
            "height": 450,
        }
        r_roads = 1
        c_roads = 2
        h_offset = 3
        w_offset = 3
        sample_cars = lambda: 0

        # sample_cars = lambda: 0
        env = CtmEnv(r_roads=r_roads,
                     c_roads=c_roads,
                     sample_cars=sample_cars,
                     render_config=render_config,
                     h_offset=h_offset,
                     w_offset=w_offset)

        # for _ in range(100):
        env.reset()
        env.render()
        for _ in range(100000):
            # sleep(0.5)
            actions = {}

            for r_road in range(r_roads):
                for c_road in range(c_roads):
                    actions["{}_{}".format(r_road, c_road)] = CHANGE
            env.step(actions)
            env.render()
        env.close()


if __name__ == '__main__':
    unittest.main()
