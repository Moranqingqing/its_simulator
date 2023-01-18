import unittest
import os
from time import sleep

from wolf.world.environments.ctm.ctm_env import CtmEnv, CHANGE, EXTEND, RED, GREEN, NORTH, WEST, SOUTH, EAST
import numpy as np


class Test(unittest.TestCase):

    def test(self):
        np.random.seed(10)
        r_roads = 1
        c_roads = 1
        H_OFFSET = 2
        W_OFFSET = 2
        render_config = {
            "width": 500,
            "height": 500,
        }
        W, H = CtmEnv.get_grid_shape(r_roads, c_roads, h_offset=H_OFFSET, w_offset=W_OFFSET)
        init_cars = np.zeros((W, H))
        # init_cars[H_OFFSET + 1][0] = 25
        init_cars[0:H_OFFSET, W_OFFSET ] = 25
        init_cars[H_OFFSET + 1:, W_OFFSET ] = 25
        # init_cars[H_OFFSET+2:][W_OFFSET] = 25
        # init_cars[:, W_OFFSET + 1] = 25
        # sample_cars = lambda: 0 if np.random.random() > 0.5 else np.random.randint(1, 25)

        init_phases = [
            [0]
        ]
        sample_cars = lambda: 0
        env = CtmEnv(r_roads=r_roads, c_roads=c_roads,
                     initial_cars=init_cars,
                     sample_cars=sample_cars,
                     h_offset=H_OFFSET,
                     w_offset=W_OFFSET,
                     initial_phases=init_phases,
                     render_config=render_config,
                     freeze_cardinals=[NORTH,WEST])

        # for _ in range(100):
        env.reset()
        env.render()
        for _ in range(100000):
            actions = {}
            sleep(1)
            for r_road in range(r_roads):
                for c_road in range(c_roads):
                    actions["{}_{}".format(r_road, c_road)] = EXTEND
            env.step(actions)
            env.render()
        env.close()


if __name__ == '__main__':
    unittest.main()
