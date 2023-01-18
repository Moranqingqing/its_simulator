from copy import copy
import gym
from gym import wrappers
from gym.spaces import Tuple, Dict, Discrete, Box, MultiDiscrete
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.qmix.qmix_policy import ENV_STATE
from numpy import binary_repr

# 25
ROAD_CELL = 0
INTERSECTION = 1
VOID = -1

RED = -1
NO_TL = 0
GREEN = 1
YELLOW = 2
ORANGE = 3

CHANGE = 1
EXTEND = 0

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3


class CtmEnv(MultiAgentEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    reward_range = (-float('inf'), float('inf'))
    spec = None

    @staticmethod
    def create_env(cls, *args, **kwargs):
        env = cls(*args, **kwargs)
        if env.agent_config["group_agents"]:
            group = {
                "unique_group": copy(env.agents)
            }
            env = env.with_agent_groups(group, obs_space=env.obs_space, act_space=env.action_space)
        return env

    @staticmethod
    def get_grid_shape(r_roads, c_roads, h_offset, w_offset):
        H = (r_roads * 2 + (r_roads + 1) * h_offset)
        W = (c_roads * 2 + (c_roads + 1) * w_offset)
        return H, W

    def __init__(self, r_roads=3, c_roads=4, h_offset=10, w_offset=10, initial_cars=None, initial_phases=None,
                 local_obs_radius=5, agent_config=None, horizon=500, render_steps=False, sample_cars=None,
                 render_config=None, freeze_cardinals=[], cell_max_capacity=25, max_crossing_cars=5,
                 min_green_time=1, max_green_time=None, reward_type="occupation", video_dir=None,
                 compressed_cars_state=False):
        if max_green_time is None:
            max_green_time = horizon
        self.radius = local_obs_radius
        self.reward_type = reward_type
        self.compressed_cars_state = compressed_cars_state
        self.video_dir = video_dir
        if local_obs_radius > (h_offset + 1) or local_obs_radius > (w_offset + 1):
            raise Exception("Radius must be lower than h_offset+1 or w_offset+1")
        self.horizon = horizon
        self.c_roads = c_roads
        self.r_roads = r_roads
        self.render_steps = render_steps

        if min_green_time > max_green_time:
            raise Exception("min_green_time should be lower than max_green_time")
        if min_green_time < 1:
            min_green_time = 1
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time

        self.cell_max_capacity = cell_max_capacity
        self.max_crossing_cars = max_crossing_cars
        self.freeze_cardinals = freeze_cardinals
        if render_config is None:
            render_config = {"width": 600, "height": 600, "debug": False}
        self.render_config = render_config
        if sample_cars is None:
            sample_cars = lambda x, direction, t: int(cell_max_capacity * 0.25)
        self.sample_cars = sample_cars

        self.H, self.W = CtmEnv.get_grid_shape(r_roads, c_roads, h_offset, w_offset)
        self.world = np.full(fill_value=VOID, shape=(self.H, self.W))
        self.traffic_lights = None
        self.cars = None
        self.h_offset = h_offset
        self.w_offset = w_offset
        self.initial_phases = initial_phases
        offset = h_offset

        for _ in range(r_roads):
            self.world[offset:offset + 2, :] = ROAD_CELL
            offset = offset + 2 + h_offset
        offset = w_offset
        for _ in range(c_roads):
            self.world[:, offset:offset + 2] = ROAD_CELL
            offset = offset + 2 + w_offset
        self.road_cell_number = 2 * w_offset * (c_roads + 1) * r_roads + 2 * h_offset * (r_roads + 1) * c_roads + 4 * (
                c_roads * r_roads)
        self.initial_cars = initial_cars

        self.viewer = None
        self.state = None
        self.agent_1 = 0
        self.agent_2 = 1
        self.k = 0
        self.window = None

        if agent_config is None:
            agent_config = {
                "type": "single-agent",  # muti-agent
                "group_agents": False,
                "default_policy": None,
                "with_env_state": False
            }

        self.agent_config = agent_config
        self.n_intersections = self.r_roads * self.c_roads
        if self.compressed_cars_state:
            cars_space = Box(0, np.inf, shape=(self.road_cell_number,))
        else:
            cars_space = Box(0, np.inf, shape=(self.H, self.W))
        global_obs_space = Dict({
            "cars": cars_space,
            "phases_ids": Box(0, len(CtmEnv.PHASES), shape=(self.r_roads, self.c_roads)),
            "green_time_progresses": Box(-2., 1., shape=(self.r_roads, self.c_roads))
        })

        local_act_space = Discrete(2)
        global_act_space = Discrete(2 ** self.n_intersections)
        grouped_act_space = Tuple([local_act_space for _ in range(self.n_intersections)])

        if self.compressed_cars_state:
            self.compressed_local_size = (self.radius - 1) * 8 + 4
            local_car_space = Box(0, np.inf, shape=(self.compressed_local_size,))
        else:
            local_car_space = Box(0, np.inf, shape=(self.radius * 2, self.radius * 2))

        local_obs_space = Dict({
            "cars": local_car_space,
            "phases_id": Box(0, len(CtmEnv.PHASES), shape=(1,)),
            "green_time_progress": Box(-2., 1., shape=(1,))
        })

        default_policy_cls = None
        default_policy_params = {}
        if agent_config["default_policy"]:
            from wolf.utils.configuration.registry import R
            default_policy_cls = R.policy_class(agent_config["default_policy"]["name"])
            default_policy_params = agent_config["default_policy"]["params"]
        if agent_config["type"] == "single-agent":
            self.agents = None
            multiagent_config = {
                "policies": {
                    "unique_policy": (default_policy_cls, global_obs_space, global_act_space, default_policy_params)},
                "policy_mapping_fn": lambda agent_name: "unique_policy"}
            self.action_space = global_act_space
            self.observation_space = global_obs_space
        elif agent_config["type"] == "multi-agent":
            self.agents = []
            for r_road in range(self.r_roads):
                for c_road in range(self.c_roads):
                    self.agents.append((r_road, c_road))
            if self.agent_config["group_agent"]:
                self.action_space = grouped_act_space

                spaces = []
                for _ in self.agents:
                    if "with_env_state" in agent_config and agent_config["with_env_state"]:
                        obs_space = Dict({"obs": local_obs_space, ENV_STATE: global_obs_space})
                    else:
                        obs_space = Dict({"obs": local_obs_space})
                    spaces.append(obs_space)
                self.obs_space = Tuple(spaces)
                self.observation_space = self.obs_space
                multiagent_config = None
            else:
                policies = {}
                if "with_env_state" in agent_config and agent_config["with_env_state"]:
                    obs_space = Dict({"obs": local_obs_space, ENV_STATE: global_obs_space})
                else:
                    obs_space = Dict({"obs": local_obs_space})
                for agent in self.agents:
                    policies[self.agent_to_str(agent)] = (
                        default_policy_cls, obs_space, local_act_space, default_policy_params)
                multiagent_config = {
                    "policies": policies,
                    "policy_mapping_fn": lambda x: x}
                self.action_space = None
                self.observation_space = obs_space
        else:
            raise Exception("agent_config[\"type\"] must be single-agent or multi-agent")

        self.multi_agent_config = multiagent_config

        self.enable = True  # TODO must be it to simulate gym env
        if self.video_dir:
            self.rec = VideoRecorder(env=self, path=self.video_dir + ".mp4")

    PHASES = [
        (GREEN, RED, GREEN, RED),
        (YELLOW, RED, YELLOW, RED),
        (ORANGE, RED, ORANGE, RED),
        (RED, RED, RED, RED),
        (RED, GREEN, RED, GREEN),
        (RED, YELLOW, RED, YELLOW),
        (RED, ORANGE, RED, ORANGE),
        (RED, RED, RED, RED)
    ]

    def reset(self):
        self.k = 0
        # self.cycles = [[None] * self.c_roads] * self.r_roads
        self.phases_ids = np.full(fill_value=-1, shape=(self.r_roads, self.c_roads), dtype=np.int8)
        self.last_green_phases_ids = np.full(fill_value=-1, shape=(self.r_roads, self.c_roads), dtype=np.int8)
        self.traffic_lights = np.full(fill_value=NO_TL, shape=(self.H, self.W), dtype=np.int8)
        self.green_times = np.full(fill_value=-1, shape=(self.r_roads, self.c_roads))
        self.greens_progession = np.full(fill_value=-1., shape=(self.r_roads, self.c_roads))
        for r_road in range(self.r_roads):
            for c_road in range(self.c_roads):
                # r, c = self.node_coordinate(r_road, c_road)
                if self.initial_phases is not None:
                    i = self.initial_phases[r_road][c_road]
                else:
                    i = np.random.randint(0, len(CtmEnv.PHASES))
                for _ in range(i + 1):
                    phase = self._next_phase(r_road, c_road)

        shape = (self.H, self.W)
        if self.initial_cars is not None:
            if self.initial_cars.shape != shape:
                raise Exception("Wrong shape : {} need={}".format(self.initial_cars.shape, shape))
            self.cars = np.array(self.initial_cars)
        else:
            self.cars = np.zeros(shape=shape)
        if not self.video_dir and self.render_steps:
            self.render()
        if self.video_dir:
            self.rec.capture_frame()
        return self.obs()

    def agent_to_str(self, agent):
        return "{}_{}".format(*agent)

    def close(self):
        if self.viewer is not None:
            self.viewer.window.close()
            self.viewer = None

    def are_cars_crossing(self, r_road, c_road):
        r, c = self.node_coordinate(r_road, c_road)
        node_cars = self.cars[r][c] + self.cars[r + 1][c] + self.cars[r][c + 1] + self.cars[r + 1][
            c + 1]
        jammed = node_cars > 0
        return jammed

    def compute_green_progression(self):
        # 0 means CHANGE will be discarded
        # 1 mean  EXTEND will be discarded
        for r in range(self.r_roads):
            for c in range(self.c_roads):
                # print("------ self.green_times[r][c]= ", self.green_times[r][c],"------------")
                abs_green_time = self.green_times[r][c]
                if abs_green_time == -1.:
                    self.greens_progession[r][c] = -2.
                elif abs_green_time < self.min_green_time:
                    prog_to_min = abs_green_time / self.min_green_time
                    self.greens_progession[r][c] = -1. + prog_to_min
                    # print("prog_to_min : ", prog_to_min)
                elif abs_green_time == self.min_green_time:
                    # could have been done in the above statement, but it is here for readbility
                    self.greens_progession[r][c] = 0.
                    # print("min reached, you change use CHANGE")
                elif abs_green_time > self.min_green_time and abs_green_time < self.max_green_time:
                    prog_to_max = (abs_green_time - self.min_green_time) / (
                            self.max_green_time - self.min_green_time)
                    self.greens_progession[r][c] = 0. + prog_to_max
                    # print("prog_to_max : ", prog_to_max)
                elif abs_green_time == self.max_green_time:
                    # could have been done in the above statement, but it is here for readbility
                    self.greens_progession[r][c] = 1.
                    # print("max reached, EXTEND will be discarded")
                else:
                    raise Exception("impossiiiibrrruuu")
        # print("green_progressions:", self.greens_progession, "------->", self.green_times)

    def step(self, action_dict):
        self.update_tl(action_dict)

        self.move_cars()

        self.k += 1
        done = self.k >= self.horizon
        dones = {"__all__": done}
        infos = {}
        if self.render_steps == True:  # in case render_steps is a fucking tuple
            self.render()
        occupation = self.occupation_metric()
        queue_lenght = self.queue_lenght_metric()
        release_at_outbounds = self.release_at_outbounds_metric()
        intersections_releases = self.release_at_intersections_metric()
        metrics = {
            "occupation": occupation,
            "queue_length": queue_lenght,
            "outbound_releases": release_at_outbounds,
            "intersections_releases": intersections_releases,
        }

        # print(queue_lenght)

        if self.agent_config["type"] == "single-agent":
            infos["single-agent"] = metrics
        else:
            for agent in self.agents:
                infos[self.agent_to_str(agent)] = metrics

        if self.video_dir:
            self.rec.capture_frame()

        if not self.video_dir and self.render_steps:
            self.render()

        return self.obs(), self.rewards(metrics[self.reward_type]), dones, infos

    def release_at_outbounds_metric(self):
        """
        Count the number of cars at the outbound cell of each horizontal roads and vertical roads in both directions.
        :return:
        """
        outbound_cells = []
        for r_road in range(self.r_roads):
            r = self.r_road_coordinate(r_road)
            outbound_cells.append((r, 0))  # WEST
            outbound_cells.append((r + 1, self.W - 1))  # EAST

        for c_road in range(self.c_roads):
            c = self.r_road_coordinate(c_road)
            outbound_cells.append((0, c + 1))  # NORTH
            outbound_cells.append((self.H - 1, c))  # SOUTH
        realeased = 0
        for cell in outbound_cells:
            realeased += self.cars[cell[0]][cell[1]]
        realeased /= (2 * self.c_roads + 2 * self.r_roads)
        return realeased

    def release_at_intersections_metric(self):
        """
        Count the number of cars in the cell directly after the intersection, mean over 4 cardinals and all intersections
        :return:
        """
        release = 0
        for r_road in range(self.r_roads):
            for c_road in range(self.c_roads):
                r = self.r_road_coordinate(r_road)
                c = self.r_road_coordinate(c_road)
                release += self.cars[r:r + 2, c:c + 2]  # compute cars on the intersections
                # for release_point in [(r - 1, c + 1), (r + 1, c + 2), (r + 2, c), (r, c - 1)]:  # north east south west
                #     release += self.cars[release_point[0]][release_point[1]]
        release /= (self.r_roads * self.c_roads * 4)
        return release

    def occupation_metric(self):
        """
        Count the number of cars on the grid. Normalised by the max_cell_capacity and number of cells.
        Can be greater than zero since gen_cell can queue up to infinity.
        :return:
        """
        car_numbers = np.sum(self.cars)
        occupation = car_numbers / (self.road_cell_number * self.cell_max_capacity)
        return -occupation

    def queue_lenght_metric(self):
        """
        Count the number of cars stopped or slowed at a traffic light. Slow means the pressure is greater than the TL capacity to realease.
        :return:
        """
        # print("------------------")
        queue_lenght = 0.
        for r_road in range(self.r_roads):
            for c_road in range(self.c_roads):
                intersection_queue = 0.
                # print(r_road, c_road)
                nesw_phase = self._get_phase(r_road, c_road)
                init_r = self.r_road_coordinate(r_road)
                init_c = self.c_road_coordinate(c_road)

                for i_cardinal, cardinal in enumerate([NORTH, EAST, SOUTH, WEST]):
                    # print(cardinal)
                    offset = self.w_offset if cardinal == EAST or cardinal == WEST else self.h_offset
                    if cardinal == NORTH:
                        delta_r = 1
                        delta_c = 0
                        r = init_r + 2
                        c = init_c + 1
                    elif cardinal == SOUTH:
                        delta_r = -1
                        delta_c = 0
                        r = init_r - 1
                        c = init_c
                    elif cardinal == EAST:
                        delta_r = 0
                        delta_c = -1
                        r = init_r + 1
                        c = init_c - 1
                    elif cardinal == WEST:
                        delta_r = 0
                        delta_c = 1
                        c = init_c + 2
                        r = init_r
                    else:
                        raise Exception("improssibruuu")
                    # we compute the queue only if cars are stopped, meaning light is not green,
                    # or light is green but traffic in jammed after the intersection

                    traffic_is_jammed_after = False
                    traffic_is_slow = False
                    if nesw_phase[i_cardinal] == GREEN:
                        traffic_is_jammed_after = self.cars[r - 3 * delta_r][c - 3 * delta_c] >= self.cell_max_capacity
                        traffic_is_slow = self.cars[r][c] >= self.max_crossing_cars

                    compute_queue = nesw_phase[i_cardinal] != GREEN or traffic_is_jammed_after or traffic_is_slow
                    # if c_road == 0 and cardinal == EAST:
                    #     print("init",init_r,init_c)
                    #     print("starting",r,c)
                    cardinal_queue = 0.
                    if compute_queue:
                        # computing queue by backpropagating starting from (r,c)
                        for idx_backprop in range(offset):
                            # divide to compute stacking queue on gen_cell (with infinite_queue)
                            cardinal_queue += self.cars[r][c] / self.cell_max_capacity
                            max_capacity = self.cars[r][c] >= self.cell_max_capacity
                            r += delta_r
                            c += delta_c
                            if not max_capacity:
                                # queue stopped
                                break
                    # if c_road==0 and cardinal==EAST:
                    #     print(r,c)
                    #     print(compute_queue)
                    #     print("cardinal_queue",cardinal_queue)
                    intersection_queue += cardinal_queue
                queue_lenght += intersection_queue
        queue_lenght /= (self.r_roads * self.c_roads * 4)  # 4 queue by intersection
        return -queue_lenght

    def get_local_cars(self, r_road, c_road):
        r, c = self.node_coordinate(r_road, c_road)
        r = r + 1  # get the middle of the intersection
        c = c + 1
        return self.cars[r - self.radius:r + self.radius, c - self.radius:c + self.radius]

    def get_local_world(self, r_road, c_road):
        r, c = self.node_coordinate(r_road, c_road)
        r = r + 1  # get the middle of the intersection
        c = c + 1
        return self.world[r - self.radius:r + self.radius, c - self.radius:c + self.radius]

    def rewards(self, global_reward):
        rewards = {}
        if self.agent_config["type"] == "single-agent":
            rewards["single-agent"] = global_reward
        else:
            for agent in self.agents:
                rewards[self.agent_to_str(agent)] = global_reward
        return rewards

    def get_local_observation(self, r_road, c_road):
        if self.compressed_cars_state:
            local_cars_ = self.get_local_cars(r_road, c_road)
            local_world = self.get_local_world(r_road, c_road)
            local_cars = np.zeros((self.compressed_local_size,))
            i = 0
            for rr in range(local_world.shape[0]):
                for cc in range(local_world.shape[1]):
                    if local_world[rr][cc] == ROAD_CELL:
                        local_cars[i] = local_cars_[rr][cc]
                        i += 1
        else:
            local_cars = self.get_local_cars(r_road, c_road)
        state = {
            "cars": np.array(local_cars),  # copy it
            "phases_id": np.array([self.phases_ids[r_road][c_road]]),
            "green_time_progress": np.array([self.greens_progession[r_road][c_road]])
        }
        return state

    def obs(self):
        # produce a value between -1 and 1, where -1 0 represent percentage of progresson toward the min
        # and 0 1 represent percentage of progression toward the max
        self.compute_green_progression()

        if self.compressed_cars_state:
            global_cars = np.zeros((self.road_cell_number,))
            i = 0
            for r in range(self.H):
                for c in range(self.W):
                    if self.world[r][c] == ROAD_CELL:
                        global_cars[i] = self.cars[r][c]
                        i += 1
        else:
            global_cars = np.array(self.cars)

        state = {
            "cars": global_cars,  # copy it
            "phases_ids": np.array(self.phases_ids),
            "green_time_progresses": np.array(self.greens_progession)
        }

        obs = {}
        if self.agent_config["type"] == "single-agent":
            obs["single-agent"] = state
        else:
            for agent in self.agents:
                r_road, c_road = agent
                local_obs = self.get_local_observation(r_road, c_road)
                if "with_env_state" in self.agent_config and self.agent_config["with_env_state"]:
                    obs[self.agent_to_str(agent)] = {
                        "obs": local_obs,
                        ENV_STATE: state
                    }
                else:
                    obs[self.agent_to_str(agent)] = {
                        "obs": local_obs
                    }
        return obs

    def _next_phase(self, r_road, c_road):
        current_phase_id = self.phases_ids[r_road][c_road]
        next_phase_id = (self.phases_ids[r_road][c_road] + 1) % len(CtmEnv.PHASES)
        self.phases_ids[r_road][c_road] = next_phase_id
        self._set_phase(r_road, c_road, next_phase_id)
        previous_phase = CtmEnv.PHASES[current_phase_id]

        if GREEN in previous_phase:
            self.last_green_phases_ids[r_road][c_road] = current_phase_id
        next_phase = CtmEnv.PHASES[next_phase_id]

        if GREEN in next_phase:
            self.green_times[r_road][c_road] = 0
        else:
            self.green_times[r_road][c_road] = -1

    def update_tl(self, action_dict):
        for c_road in range(self.c_roads):
            for r_road in range(self.r_roads):
                phase = CtmEnv.PHASES[self.phases_ids[r_road][c_road]]
                # the agent can interact with it if and only if phase is GREEN
                if GREEN in phase:
                    if self.agent_config["type"] == "single-agent":
                        node_action = action_dict["single-agent"]
                        node_action = binary_repr(node_action, width=self.n_intersections)
                        node_action = int(node_action[c_road + self.c_roads * r_road])
                    else:
                        node_action = action_dict[self.agent_to_str((r_road, c_road))]
                    if node_action == EXTEND:
                        if self.green_times[r_road][c_road] >= self.max_green_time:
                            # forcing change
                            self._next_phase(r_road, c_road)
                        else:
                            pass
                    elif node_action == CHANGE:
                        if self.green_times[r_road][c_road] < self.min_green_time:
                            # forcing extend
                            pass
                        else:
                            self._next_phase(r_road, c_road)
                    else:
                        raise Exception("Unknow action : {}".format(node_action))

                else:
                    if YELLOW in phase or ORANGE in phase:
                        self._next_phase(r_road, c_road)
                    elif phase == (RED, RED, RED, RED):
                        if self.are_cars_crossing(r_road, c_road):
                            # cars are still crossing the intersection, so all tl must stay red
                            print(
                                "[WARNING] this should not happen since cars are not supposed to cross if they can be stuck in the node (unless init_cars put cars on nodes)")
                            pass
                        else:
                            self._next_phase(r_road, c_road)
                    else:
                        raise Exception("Impossible")

                if GREEN in CtmEnv.PHASES[self.phases_ids[r_road][c_road]]:
                    self.green_times[r_road][c_road] += 1

    def move_cars(self):
        self.previous_cars = self.cars
        self.cars = np.zeros(self.previous_cars.shape)

        for cardinal in [EAST, SOUTH, WEST, NORTH]:
            # self.render()
            if cardinal not in self.freeze_cardinals:
                # The following code is written for car going to east.
                # So for each horizontal roads, we backprop cars from outer cell to inner cell.
                # the whole world is rotating 90° after this, in order to apply the same process to all direction (north, south and west)
                # Each cell follow a specific pattern.
                # cell0 is the cell directly on the left to the node
                # cell1 is the left cell of the node
                # cell2 is the right cell of the node
                # cell3 is the cell directly on the right to the node
                # gen_cell is the far left cell, the one that generates cars
                # basic_cell denotes all other cells

                for r_road in range(self.r_roads):

                    r = self.r_road_coordinate(r_road)
                    r = r + 1
                    jam_on_the_right = 0
                    for c_ in range(self.cars.shape[1]):

                        c = self.W - c_ - 1

                        curr_cell_is_tl = self.traffic_lights[r][c] != NO_TL
                        curr_cell_is_green = self.traffic_lights[r][c] == GREEN

                        curr_cell_is_not_tl = self.traffic_lights[r][c] == NO_TL
                        prev_cell_exists = c - 1 >= 0
                        pprev_cell_exists = c - 2 >= 0
                        prev_cell_is_not_tl = prev_cell_exists and self.traffic_lights[r][c - 1] == NO_TL
                        prev_cell_is_tl = prev_cell_exists and self.traffic_lights[r][c - 1] != NO_TL
                        next_cell_exists = c + 1 < self.W
                        next_cell_is_tl = next_cell_exists and self.traffic_lights[r][c + 1] != NO_TL
                        next_cell_is_green = next_cell_exists and self.traffic_lights[r][c + 1] == GREEN
                        next_cell_is_not_tl = next_cell_exists and self.traffic_lights[r][c + 1] == NO_TL

                        prev_cell_is_red = prev_cell_exists and self.traffic_lights[r][c - 1] == RED
                        prev_cell_is_green = prev_cell_exists and self.traffic_lights[r][c - 1] == GREEN

                        pprev_cell_is_red = pprev_cell_exists and self.traffic_lights[r][c - 2] == RED

                        cell_type = None
                        other_cardinal_should_process_this_cell = False
                        if prev_cell_is_not_tl and curr_cell_is_not_tl and c > 0 and (
                                not (next_cell_exists) or (next_cell_exists and next_cell_is_not_tl)):
                            cell_type = "basic_cell"
                        elif c == 0:
                            cell_type = "gen_cell"
                        elif curr_cell_is_not_tl and next_cell_is_tl:
                            cell_type = "cell0"
                        elif curr_cell_is_not_tl and prev_cell_is_tl:
                            cell_type = "cell3"
                        elif curr_cell_is_tl and prev_cell_is_tl:
                            cell_type = "cell2"
                        else:
                            cell_type = "cell1"

                        if cell_type in ["cell1", "cell2", "cell3"]:
                            c_road = int(np.round(c / (self.w_offset + 2))) - 1
                            tl_was_green = CtmEnv.PHASES[self.last_green_phases_ids[r_road][c_road]][cardinal] == GREEN

                        if cell_type == "basic_cell":
                            # add cars from the left
                            self.cars[r][c] += self.previous_cars[r][c - 1]
                        elif cell_type == "gen_cell":
                            self.cars[r][c] = self.sample_cars(
                                c_road if (cardinal == NORTH or cardinal == SOUTH) else r_road, cardinal, self.k)
                        elif cell_type == "cell0":
                            # add cars from the left
                            self.cars[r][c] += self.previous_cars[r][c - 1]
                            tl_on_the_right = self.traffic_lights[r][c + 1]
                            if tl_on_the_right == RED or tl_on_the_right == ORANGE or tl_on_the_right == YELLOW:
                                self.cars[r][c] += self.previous_cars[r][c]
                            else:
                                if unmovable_from_cell0 > 0:
                                    self.cars[r][c] += unmovable_from_cell0
                            unmovable_from_cell0 = None
                        elif cell_type == "cell1":
                            unmovable_from_cell0 = 0
                            if curr_cell_is_green:
                                space_remaining_on_next_cells = 1 * self.cell_max_capacity - (
                                        self.cars[r][c + 1] + self.cars[r][c + 2])  # + self.cars[r][c + 3])
                                movable_cars = min(self.max_crossing_cars, space_remaining_on_next_cells)
                                # movable_cars = 2 * CELL_MAX_CAPACITY - (self.cars[r][c + 1] + self.cars[r][c + 2] + self.cars[r][c + 3])
                                if movable_cars > 0:
                                    if movable_cars >= self.previous_cars[r][c - 1]:
                                        self.cars[r][c] += self.previous_cars[r][c - 1]
                                    else:
                                        self.cars[r][c] += movable_cars
                                unmovable_from_cell0 = self.previous_cars[r][c - 1] - movable_cars
                            elif tl_was_green and not curr_cell_is_green and not next_cell_is_green:
                                self.cars[r][c] = 0
                            else:
                                # dont do anything, another cardinal have to take care of this
                                other_cardinal_should_process_this_cell = True
                                pass  # self.cars[r][c] += self.previous_cars[r][c]
                        elif cell_type == "cell2":
                            jammed = (tl_was_green and prev_cell_is_red and not curr_cell_is_green)
                            can_move_cars = jammed or not prev_cell_is_red
                            if can_move_cars:
                                self.cars[r][c] += self.previous_cars[r][c - 1]
                            else:
                                # dont do anything, another cardinal have to take care of this
                                other_cardinal_should_process_this_cell = True
                                pass  # self.cars[r][c] += self.previous_cars[r][c]
                        elif cell_type == "cell3":
                            jammed = (tl_was_green and not prev_cell_is_green)
                            if jammed or not pprev_cell_is_red:
                                self.cars[r][c] += self.previous_cars[r][c - 1]
                        else:
                            pass

                        if not other_cardinal_should_process_this_cell:
                            self.cars[r][c] += jam_on_the_right
                            if cell_type != "gen_cell":
                                # compute the extra cars that can't fit on this cell
                                jam_on_the_right = self.cars[r][c] - self.cell_max_capacity
                                if jam_on_the_right > 0:
                                    self.cars[r][c] = self.cell_max_capacity
                                else:
                                    jam_on_the_right = 0
                            else:
                                # generator cell accumulates all traffic jam
                                jam_on_the_right = 0
                        else:
                            # if cell is moving in another direction, they should be no traffic jam propagation is in the current direction
                            jam_on_the_right = 0
            # self.render()
            # rotate the grid
            self.cars = np.rot90(self.cars, k=1, axes=(0, 1))
            self.previous_cars = np.rot90(self.previous_cars, k=1, axes=(0, 1))
            self.traffic_lights = np.rot90(self.traffic_lights, k=1, axes=(0, 1))
            self.world = np.rot90(self.world, k=1, axes=(0, 1))  # only usefull for rendering
            self.phases_ids = np.rot90(self.phases_ids, k=1, axes=(0, 1))
            self.last_green_phases_ids = np.rot90(self.last_green_phases_ids, k=1, axes=(0, 1))
            tmp = self.W
            self.W = self.H
            self.H = tmp
            tmp = self.w_offset
            self.w_offset = self.h_offset
            self.h_offset = tmp
            tmp = self.r_roads
            self.r_roads = self.c_roads
            self.c_roads = tmp

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        
        HEIGHT = self.render_config["height"]
        WIDTH = self.render_config["width"]

        cell_height = HEIGHT / self.H
        cell_witdh = WIDTH / self.W

        if self.viewer is None:
            self.viewer = rendering.Viewer(WIDTH, HEIGHT)

        for horizontal_line in range(self.H):
            self.viewer.draw_line((0, horizontal_line * cell_height), (WIDTH, horizontal_line * cell_height),
                                  color=(0.0, 0.0, 0.0))
        for vertical_line in range(self.W):
            self.viewer.draw_line((vertical_line * cell_witdh, 0), (vertical_line * cell_witdh, HEIGHT),
                                  color=(0.0, 0.0, 0.0))

        def draw_cell(r, c, color):
            x = c * cell_witdh
            y = HEIGHT - r * cell_height

            x_ = (c + 1) * cell_witdh
            y_ = HEIGHT - (r + 1) * cell_height
            self.viewer.draw_polygon([(x, y), (x, y_), (x_, y_), (x_, y)], color=color)

        def draw_tl(r, c, light, cardinal, start_pos=0, end_pos=0.25):
            if light == GREEN:
                color = (0., 1., 0)
            elif light == RED:
                color = (1., 0., 0.)
            elif light == YELLOW:
                color = (1., 1.0, 0.)
            elif light == ORANGE:
                # color = (1., 0.50, 0.)
                color = (1., 0.5, 0.)
            else:
                return

            if cardinal == WEST:
                x = (c + 2 - end_pos) * cell_witdh
                y = HEIGHT - (r + 0) * cell_height
                x_ = (c + 2 - start_pos) * cell_witdh
                y_ = HEIGHT - (r + 1.0) * cell_height
            elif cardinal == EAST:
                x = (c + start_pos) * cell_witdh
                y = HEIGHT - (r + 1.0) * cell_height
                x_ = (c + end_pos) * cell_witdh
                y_ = HEIGHT - (r + 2.0) * cell_height
            elif cardinal == NORTH:
                x = (c + 1) * cell_witdh
                y = HEIGHT - (r + 2 - end_pos) * cell_height
                x_ = (c + 2) * cell_witdh
                y_ = HEIGHT - (r + 2 - start_pos) * cell_height
            elif cardinal == SOUTH:
                x = (c + 0) * cell_witdh
                y = HEIGHT - (r + start_pos) * cell_height
                x_ = (c + 1) * cell_witdh
                y_ = HEIGHT - (r + end_pos) * cell_height
            else:
                raise Exception()

            self.viewer.draw_polygon([(x, y), (x, y_), (x_, y_), (x_, y)], color=color)

        for r in range(self.H):
            for c in range(self.W):
                if self.world[r][c] == ROAD_CELL:
                    n_cars = self.cars[r][c]

                    # if self.render_config["debug"] and n_cars == CELL_MAX_CAPACITY:
                    #     draw_cell(r, c, color=(.7, 0., .7 ))
                    if n_cars == 0:
                        draw_cell(r, c, color=(0, 0, 0))
                    elif n_cars <= self.cell_max_capacity:
                        draw_cell(r, c, color=(0., .2, 1. * (n_cars / self.cell_max_capacity)))
                    else:
                        draw_cell(r, c, color=(0.5, .2, 1.))

        for c_road in range(self.c_roads):
            for r_road in range(self.r_roads):
                previous_phase = CtmEnv.PHASES[self.phases_ids[r_road][c_road] - 1 % len(CtmEnv.PHASES)]
                pprevious_phase = CtmEnv.PHASES[self.phases_ids[r_road][c_road] - 2 % len(CtmEnv.PHASES)]
                previous_green_phase = CtmEnv.PHASES[self.last_green_phases_ids[r_road][c_road]]
                r, c = self.node_coordinate(r_road, c_road)
                for cardinal in [EAST, NORTH, SOUTH, WEST]:
                    draw_tl(r, c, self._get_light(r, c, cardinal), cardinal, 0.0, 0.25)
                    if self.render_config["debug"]:
                        draw_tl(r, c, previous_phase[cardinal], cardinal, 0.30, 0.35)
                        draw_tl(r, c, pprevious_phase[cardinal], cardinal, 0.40, 0.45)
                        draw_tl(r, c, previous_green_phase[cardinal], cardinal, 0.60, 0.65)

        for horizontal_line in range(self.H):
            self.viewer.draw_line((0, horizontal_line * cell_height), (WIDTH, horizontal_line * cell_height),
                                  color=(0.0, 0.0, 0.0))
        for vertical_line in range(self.W):
            self.viewer.draw_line((vertical_line * cell_witdh, 0), (vertical_line * cell_witdh, HEIGHT),
                                  color=(0.0, 0.0, 0.0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        # return self.viewer.render(return_rgb_array = True)

    def node_coordinate(self, r_road, c_road):
        r = self.r_road_coordinate(r_road)
        c = self.c_road_coordinate(c_road)
        return r, c

    def r_road_coordinate(self, r_road):
        return (r_road + 1) * (self.h_offset) + (r_road) * 2

    def c_road_coordinate(self, c_road):
        return (c_road + 1) * (self.w_offset) + (c_road) * 2

    def _set_phase(self, r_road, c_road, phase_id):
        r, c = self.node_coordinate(r_road, c_road)
        n, e, s, w = CtmEnv.PHASES[phase_id]
        self._set_light(r, c, n, NORTH)
        self._set_light(r, c, e, EAST)
        self._set_light(r, c, s, SOUTH)
        self._set_light(r, c, w, WEST)

    def _get_phase(self, r_road, c_road):
        r, c = self.node_coordinate(r_road, c_road)
        n = self._get_light(r, c, NORTH)
        e = self._get_light(r, c, EAST)
        s = self._get_light(r, c, SOUTH)
        w = self._get_light(r, c, WEST)
        return n, e, s, w

    def _get_light(self, r, c, cardinal):
        if cardinal == NORTH:
            return self.traffic_lights[r + 1][c + 1]
        elif cardinal == WEST:
            return self.traffic_lights[r][c + 1]
        elif cardinal == EAST:
            return self.traffic_lights[r + 1][c]
        elif cardinal == SOUTH:
            return self.traffic_lights[r][c]
        else:
            raise Exception()

    def _set_light(self, r, c, light, cardinal):
        if cardinal == NORTH:
            self.traffic_lights[r + 1][c + 1] = light
        elif cardinal == WEST:
            self.traffic_lights[r][c + 1] = light
        elif cardinal == EAST:
            self.traffic_lights[r + 1][c] = light
        elif cardinal == SOUTH:
            self.traffic_lights[r][c] = light
        else:
            raise Exception()

    def get_cycle(self, r_road, c_road):
        # return self.cycles[r_road][c_road]
        return self.phases_ids[r_road * self.c_roads + c_road]


if __name__ == "__main__":
    np.random.seed(11)
    env_config = {
        "agent_config": {"type": "multi-agent", "default_policy": None, "group_agent": False},
        "render_config": {
            "width": 933,
            "height": 333,
            "debug": True
        },
        "reward_type": "queue_length",
        "compressed_cars_state": True

    }
    from wolf.world.environments.env_factories import *
    import time

    env = ctm_test5(env_config)
    env.render_steps = True
    # env.sample_cars = lambda x, cardinal, k: 5
    rec = VideoRecorder(env=env, path="tmp.mp4")
    env.reset()
    rec.capture_frame()

    for _ in range(1000):
        # time.sleep(1.0)
        actions = {}

        for agent in env.agents:
            actions[env.agent_to_str(agent)] = EXTEND  # EXTEND #if np.random.random() > 0.5 else CHANGE
        o_, r_, done, info = env.step(actions)
        print(info["0_0"]["queue_length"], np.sum(env.cars))
        rec.capture_frame()
    env.close()
    rec.close()
