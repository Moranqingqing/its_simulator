import unittest
import os
from time import sleep

from wolf.world.environments.ctm.ctm_env import CtmEnv, CHANGE, EXTEND, RED, GREEN
import numpy as np


class Test(unittest.TestCase):

    def test0(self):
        np.random.seed(10)
        render_config = {
            "width": 600,
            "height": 300,
        }
        r_roads = 1
        c_roads = 2
        h_offset = 3
        w_offset = 3
        sample_cars = lambda: 0

        W, H = CtmEnv.get_grid_shape(r_roads, c_roads, h_offset=h_offset, w_offset=w_offset)
        init_cars = np.zeros((W, H))
        # init_cars[H_OFFSET + 1][0] = 25
        init_cars[h_offset + 1, 0:w_offset+2] = 25
        sample_cars = lambda: 0

        init_phases = [
            [7, 7]
        ]

        # sample_cars = lambda: 0
        env = CtmEnv(r_roads=r_roads,
                     c_roads=c_roads,
                     sample_cars=sample_cars,
                     initial_cars=init_cars,
                     initial_phases=init_phases,
                     render_config=render_config,
                     h_offset=h_offset,
                     w_offset=w_offset)

        # for _ in range(100):
        env.reset()
        env.render()
        for _ in range(100000):
            sleep(1)
            actions = {}

            for r_road in range(r_roads):
                for c_road in range(c_roads):
                    actions["{}_{}".format(r_road, c_road)] = EXTEND if np.random.random() > 0.5 else CHANGE
            env.step(actions)
            # if env.cars[4][3] == 25 and  env.cars[4][4] == 25 and env.cars[4][5]==0 and
            env.render()
        env.close()


if __name__ == '__main__':
    unittest.main()
