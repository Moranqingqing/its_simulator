from collections import Iterable

import numpy as np
import random

from gym.spaces import Discrete, Box

from rltl.envs.gridworld.geometry import inRectangle, isSegmentIntersectRectangle
from rltl.envs.gridworld.noise import dynamics
from rltl.envs.gridworld.world import World


class EnvGridWorld(object):
    CARDINAL_ACTIONS = [(0., 0.), (0., 1.), (1., 0.), (-1., 0), (0., -1.)]
    CARDINAL_ACTIONS_STR = ["X", "v", ">", "<", "^"]

    def action_space(self):
        return self.actions

    def action_space_str(self):
        return self.actions_str

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # def random_state(self):
    #     return (random.random() * self.w, random.random() * self.h)

    def __init__(self,
                 dim,
                 cases,
                 horizon,
                 walls_inside=False,
                 id="default_id",
                 penalty_on_move=0, actions=None, actions_str=None,
                 init_s=(0.5, 0.5),
                 cost_on_move=0, blocks=None,
                 normalise_state=True,
                 block_by_a_wall_response="stop_just_before_intersection",
                 dynamics_params=None):
        if actions_str is None:
            actions_str = EnvGridWorld.CARDINAL_ACTIONS_STR
        if actions is None:
            actions = EnvGridWorld.CARDINAL_ACTIONS
        if blocks is None:
            blocks = []
        if dynamics_params is None:
            dynamics_params = {"type": "s+a"}
        self.normalise_state = normalise_state
        self.dynamics_params = dynamics_params
        self.block_by_a_wall_response = block_by_a_wall_response
        # random.seed(seed)
        # np.random.seed(seed)
        # self.seed = seed
        # List des actions possible et leur description textuelle
        self.actions = actions
        self.actions_str = actions_str

        # identifiant de l'environement
        self.id = id

        # taille de la grille
        w, h = dim

        wall_blocks = []

        # if walls_outside:
        #     for i in range(-1, w + 1):  # north wall
        #         wall_blocks.append((i, 0, i + 1, 1))
        #
        #     for i in range(-1, w + 1):  # south wall
        #         wall_blocks.append((i, h - 1, i + 1, h))
        #
        #     for j in range(1 - 1, h - 1 + 1):  # east wall
        #         wall_blocks.append((0, j, 1, j + 1))
        #
        #     for j in range(1 - 1, h - 1 + 1):  # west wall
        #         wall_blocks.append((w - 1, j, w, j + 1))

        if walls_inside:
            for i in range(0, w):  # north wall
                wall_blocks.append((i, 0, i + 1, 1))

            for i in range(0, w):  # south wall
                wall_blocks.append((i, h - 1, i + 1, h))

            for j in range(1, h - 1):  # east wall
                wall_blocks.append((0, j, 1, j + 1))

            for j in range(1, h - 1):  # west wall
                wall_blocks.append((w - 1, j, w, j + 1))

        self.blocks = blocks + wall_blocks

        faulty_rectangle = []
        for case in blocks + wall_blocks + cases:
            if type(case[0]) == tuple:
                (xmin, ymin, xmax, ymax), _, _, _ = case
            else:
                xmin, ymin, xmax, ymax = case

            if xmin > xmax or ymin > ymax:
                faulty_rectangle.append(case)
        if len(faulty_rectangle) > 0:
            raise Exception("Rectangles should be described with upperleft then bottomright point: {}".format(faulty_rectangle))
        self.w = float(w)
        self.h = float(h)

        # liste des case specifiques de la grille
        self.cases = cases

        # reward/cout par defaut a chaque mouvement
        self.penalty_on_move = penalty_on_move
        self.cost_on_move = cost_on_move

        # attribut des trajectoires
        self.horizon = horizon
        self.current_case = None
        self.init_s = init_s
        # print self.cases

        self.action_space = Discrete(len(self.actions))
        if self.normalise_state:
            self.observation_space = Box(low=0., high=1., shape=(2,))
        else:
            self.observation_space = Box(low=0., high=max(w, h), shape=(2,))
        self.action_space_str = actions_str
        self.last_action = None
        self.viewer = None
        self.trajectory = []
        self.reset()

    def render(self, save_prefix=None):
        x, y = self.state
        self.world = World(self)
        self.world.draw_lattice()
        self.world.draw_frame()
        self.world.draw_cases()
        xa, ya = self.last_action
        self.world.draw_trajectory(self.trajectory, (0.5, 0.5, 0.5, 1), line_width=10)
        # drawing state
        self.world.draw_rectangle((x - 0.1, y - 0.1, x + 0.1, y + 0.1), color=(1, 1, 1))
        # drawing action
        self.world.draw_rectangle((x + xa, y + ya, x + xa + 0.1, y + ya + 0.1), color=(0, 1, 1))

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        self.viewer.imshow(self.world.to_numpy_img())

        if save_prefix is not None:
            self.world.save(save_prefix + "_" + str(len(self.trajectory) - 1))
            if self.done:
                import glob
                from PIL import Image
                # filepaths
                fp_in = save_prefix + "_*.png"
                fp_out = save_prefix + ".gif"
                img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
                img.save(fp=fp_out, format='GIF', append_images=imgs,
                         save_all=True, duration=200, loop=0)

    def reset(self):
        self.trajectory.clear()
        self.done = False
        self.state = self.init_s
        self.t = 0
        self.ended = False
        if self.normalise_state:
            observation = np.array([self.state[0] / self.w, self.state[1] / self.h])
        else:
            observation = np.array(self.state)
        return observation

    def step(self, i_a):
        a = self.actions[i_a]
        ax, ay = a
        self.last_action = a
        if self.ended:
            raise Exception('game is ended')

        x, y = self.state
        s = (x, y)
        rp = 0.
        cp = 0.

        # if 6.4<x==6.6 and 2.4<y < 2.6:
        #     print("hello")

        # checking if we are already in an absorbing case
        for case in self.cases:
            rectangle, r, c, is_absorbing = case
            if inRectangle(s, rectangle):
                if is_absorbing:
                    cp = 0.
                    rp = 0.
                    self.ended = True
                    sp = (x, y)
                    info = {"c_": cp}
                    absorbing = True
                    break

        # checking if we are already in a block
        for block in self.blocks:
            if inRectangle(s, block):
                cp = 0.
                rp = 0.
                self.ended = True
                sp = (x, y)
                info = {"c_": cp}
                absorbing = True
                break

        # if agent decides to not move, then it is done.
        if (ax == 0. and ay == 0.):
            cp = 0.
            rp = 0.
            sp = (x, y)
            absorbing = True
            self.ended = True

        if not self.ended:
            # on se deplace

            # component stuff here, instead of ugly if then else and "apply_noise"

            if not self.ended and not (ax == 0. and ay == 0.):
                xp, yp = dynamics(x, y, ax, ay, self.dynamics_params)
            else:
                xp, yp = x, y

            if (x == xp and y == yp):  # nothing happen
                rp = 0.
                cp = 0.
                sp = (x, y)
                absorbing = False
            else:
                # collisions
                walls_on_the_ways = []
                for wall in self.blocks:  # checking if agent ended up in a wall
                    if isSegmentIntersectRectangle((x, y, xp, yp), wall):  # inRectangle((xp, yp), wall):
                        walls_on_the_ways.append(wall)

                agent_left_grid = False
                if len(walls_on_the_ways) == 0:
                    agent_left_grid = xp < 0 or xp > self.w or yp < 0 or yp > self.h

                if len(walls_on_the_ways) > 0 or agent_left_grid:
                    # sp = (x, y)
                    # rp = 0
                    # cp = 0
                    # absorbing = False
                    if self.block_by_a_wall_response == "do_not_move":
                        sp = (x, y)
                        rp = 0
                        cp = 0
                        absorbing = False
                    elif self.block_by_a_wall_response == "stop_just_before_intersection":
                        # ugly as shit, probably a better method for this
                        dx = (xp - x) / 100.
                        dy = (yp - y) / 100.

                        for i in range(100):
                            xx = x + i * dx
                            yy = y + i * dy
                            if agent_left_grid:
                                out = xx < 0 or xx > self.w or yy < 0 or yy > self.h
                            else:
                                out = False
                                for wall_on_the_way in walls_on_the_ways:
                                    if inRectangle((xx, yy), wall_on_the_way):
                                        out = True
                                        break
                            xp = x + (i - 1) * dx
                            yp = y + (i - 1) * dy
                            sp = (xp, yp)
                            if out:
                                break
                        absorbing = False


                else:
                    # normal behavior
                    absorbing = False
                    sp = (xp, yp)
                    for case in self.cases:
                        rectangle, r, c, is_absorbing = case
                        if inRectangle(sp, rectangle):
                            rp = r
                            cp = c
                            absorbing = is_absorbing
                            break

            s = self.state
            self.state = sp
            self.t += 1

            info = {"c_": cp}
            info["state_is_absorbing"] = absorbing
            self.ended = self.ended or self.t >= self.horizon

            rp = rp - self.penalty_on_move

        if self.normalise_state:
            observation = np.array([sp[0] / self.w, sp[1] / self.h])
        else:
            observation = np.array(sp)

        reward, done, info = rp, self.ended, info

        self.trajectory.append((s, a, reward, sp, done, info))
        self.done = done
        return observation, reward, done, info
        # return t
