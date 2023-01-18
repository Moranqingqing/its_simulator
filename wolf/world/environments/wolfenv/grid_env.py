import pprint
import numpy as np
import logging

from flow.controllers import SimCarFollowingController, GridRouter
from flow.core.params import VehicleParams, SumoCarFollowingParams, SumoParams, EnvParams, NetParams, \
    InitialConfig, DetectorParams, InFlows
from flow.networks.asymetric_grid_network import TrafficLightAsymetricGridNetwork

from wolf.world.environments.wolfenv.wolf_env import WolfEnv
from wolf.utils.math import recursive_dict_update


class GenericGridEnv(WolfEnv):
    def __init__(
            self,
            tl_params,
            agents_params,
            group_agents_params,
            multi_agent_config_params,
            vehicles_params,
            env_params,
            sim_params,
            detector_params,
            net_params,
            initial_config_params,
            inflow_type,
            inflow_params,
            env_state_params=None,
            action_repeat_params=None,
            simulator='traci'
    ):
        self.logger = logging.getLogger(__name__)
        n_row = len(net_params["additional_params"]["grid_array"]["row_inner_lengths"]) + 1
        n_column = len(net_params["additional_params"]["grid_array"]["col_inner_lengths"]) + 1
        cars_left = 0
        cars_right = 0
        cars_top = 0
        cars_bot = 0

        detector_params_obj = DetectorParams()
        for i in range(n_row * n_column):  # TODO change 9 to n_row x n_cols ??
            detector_params_obj.add_induction_loop_detectors_to_intersection(
                name=f"det_{i}",
                node_id=f"center{i}",
                **detector_params)

        vehicles = VehicleParams()

        num_vehicles = (cars_left + cars_right) * n_column + (cars_bot + cars_top) * n_row
        vehicles.add(
            veh_id="human",
            acceleration_controller=(SimCarFollowingController, {}),
            car_following_params=SumoCarFollowingParams(**vehicles_params),
            routing_controller=(GridRouter, {}),
            num_vehicles=num_vehicles)

        initial_config = InitialConfig(**initial_config_params)

        outer_edges = []
        outer_edges += ["left{}_{}".format(n_row, column) for column in range(n_column)]
        outer_edges += ["right{}_{}".format(0, column) for column in range(n_column)]
        outer_edges += ["bot{}_{}".format(row, 0) for row in range(n_row)]
        outer_edges += ["top{}_{}".format(row, n_column) for row in range(n_row)]

        # outer_edges = [edge["id"] for edge in network._outer_edges]
        #
        # self.logger.info([edge["id"] for edge in network._outer_edges])
        self.logger.info(outer_edges)

        m = {
            "left": [],
            "right": [],
            "bot": [],
            "top": []
        }
        # pre computing the mapping to avoid doing it each time mapping function is called
        for key, schedules in m.items():
            if key == "left":
                params = inflow_params['NS']
            if key == "right":
                params = inflow_params['SN']
            if key == "bot":
                params = inflow_params['WE']
            if key == "top":
                params = inflow_params['EW']
            if params is None:
                # meaning no inflow for this direction
                continue
            else:
                p1, p2 = params
            if inflow_type == 'gaussian':
                sampling_interval = 20
                schedules.extend(self.get_gaussian_schedules(p1, p2, sampling_interval, env_params["horizon"]))
            elif inflow_type == 'platoon':
                schedules.extend(self.get_platoon_schedules(p1, p2, env_params["horizon"]))
            elif inflow_type == 'poisson':
                schedules.extend(self.get_poisson_schedules(p1, env_params["horizon"]))
            else:
                raise NotImplementedError(inflow_type)

        self.logger.info("inflow schedules:\n{}".format(pprint.pformat(m)))

        # print("inflow schedules:\n{}".format(pprint.pformat(m)))

        def mapping(edge):
            if "left" in edge:
                return m["left"]
            if "right" in edge:
                return m["right"]
            if "bot" in edge:
                return m["bot"]
            if "top" in edge:
                return m["top"]

        from wolf.world.environments.wolfenv.inflow_schedule import (generate_gaussian_inflow, generate_platoon_inflow,
                                                                     generate_poisson_inflow)
        if inflow_type == 'gaussian':
            inflow_reset_function = lambda: generate_gaussian_inflow(outer_edges, mapping)
        elif inflow_type == 'platoon':
            inflow_reset_function = lambda: generate_platoon_inflow(outer_edges, mapping)
        elif inflow_type == 'poisson':
            inflow_reset_function = lambda: generate_poisson_inflow(outer_edges, mapping)
        else:
            raise NotImplementedError(inflow_type)

        # initial_inflow = None  # TODO dont do that since it will be reseted anyway
        # initial_inflow = inflow_reset_function()

        # N = 5
        # initial_inflow = InFlows()
        # for i in range(0, N):
        #     for edge in outer_edges:
        #         # if i%2==0:
        #         initial_inflow.add(
        #             veh_type="human",
        #             edge=edge,
        #             probability=0.8,
        #             depart_lane="free",
        #             depart_speed=vehicles_params["max_speed"],
        #             begin=i * (500/N) + 1,
        #             end=(i + 1) * 500/N
        #         )
        # print(initial_inflow.get())

        network = TrafficLightAsymetricGridNetwork(
            name="taffic_light_asymetric_grid_network",
            vehicles=vehicles,
            net_params=NetParams(inflows=None, **net_params),
            initial_config=initial_config,
            detector_params=detector_params_obj
        )

        WolfEnv.__init__(
            self=self,
            agents_params=agents_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            env_params=EnvParams(**env_params),
            sim_params=SumoParams(**sim_params),
            network=network,
            tl_params=tl_params,
            inflows_generator=inflow_reset_function,
            env_state_params=env_state_params,
            controlled_nodes=None,
            action_repeat_params=action_repeat_params,
            simulator=simulator
        )

    def get_gaussian_schedules(self, mu, sigma, sampling_interval, horizon):
        """
        Creates gaussian schedules.

        Args:
            mu (float): Mean of the gaussian distribution.
            sigma (float): Std. Deviation of the gaussian distribution.
            sampling_interval (int): Sampling interval.
            horizon (int): Environment horizon.

        Returns:
            list: List of schedules: [(t0, t1, mu, sigma), (t1, t2, mu, sigma), ...]
        """
        schedules = []
        for start_time in range(0, horizon, sampling_interval):
            end_time = min(start_time + sampling_interval, horizon)
            schedule = (start_time, end_time, mu, sigma)
            schedules.append(schedule)
        return schedules

    def get_platoon_schedules(self, p_time, np_time, horizon, p_period=2, np_period=-1):
        """
        Creates platoon schedules.
        If inflow has Period X then equally spaced vehicles are inserted at interval of X seconds.
        Period value of -1 is used to denote the no flow condition.
        For more refer to flow.core.params.InFlows.

        Args:
            p_time (int): Platoon time or duration.
            np_time (int): No-Platoon time or duration.
            horizon (int): Environment horizon.
            p_period (int, optional): Platoon Period. Adds vehicles every 'p_period' seconds. Defaults to 2.
            np_period (int, optional): No-Platoon Period. Adds vehicles every 'np_period' seconds. Defaults to -1.

        Returns:
            list: List of schedules: [(t0, t1, p_period), (t1, t2, np_period), ...]
        """
        schedules = []
        interval = p_time + np_time

        for start_time in range(0, horizon, interval):
            p_end_time = min(start_time + p_time, horizon)
            if p_time > 0:
                schedule = (start_time, p_end_time, p_period)
                schedules.append(schedule)

            if p_end_time < horizon:
                np_end_time = min(p_end_time + np_time, horizon)
                schedule = (p_end_time, np_end_time, np_period)
                schedules.append(schedule)

        return schedules

    def get_poisson_schedules(self, rate, horizon):
        """
        Schedules the cars to arrive in a Poisson process of the specified rate

        Args:
            rate (float): Poisson rate of incoming cars on an edge
            horizon (int): Environment horizon
        Returns:
            list: List of floats [t0, t1, ...] of car arrivals
        """
        arrivals = []
        time = 1
        scale = 1/(rate + 1e-6)    # Exponential distribution is implemented with a scale parameter
                                   # in numpy instead of a rate parameter
        while time < horizon:
            time += np.random.exponential(scale)    # Waiting times for a Poisson process are exponential
            arrivals.append(time)
        return arrivals


class SimpleGridEnv(GenericGridEnv):
    def __init__(
            self,
            agents_params,
            group_agents_params,
            multi_agent_config_params,
            env_state_params,
            n=3,
            m=3,
            inflow_type='gaussian',
            inflow_params={'WE': (0.3, 0), 'EW': (0.3, 0), 'NS': (0.3, 0), 'SN': (0.3, 0)},
            simulator='traci',
            sim_params=None,
            horizon=500,
            row_inner_lengths=None,
            col_inner_lengths=None,
            detector_params={'positions': [-5, -100], 'frequency': 100},
            action_repeat_params=None,
            tl_params={},
            short_length=None,
            long_length=None
    ):
        if col_inner_lengths is None:
            col_inner_lengths = [] if m == 1 else (m - 1) * [300]
        if row_inner_lengths is None:
            row_inner_lengths = [] if n == 1 else (n - 1) * [300]
        if short_length is None:
            short_length = 1000
        if long_length is None:
            long_length = 1000
        tl_params_default = {
            "ALL": {
                "name": "second_based",
                "params": {
                    "phases": [
                        {"colors": "rGrG", "min_time": 10, "max_time": 60},  # "inf"
                        {"colors": "GrGr", "min_time": 10, "max_time": 60},  # "inf"
                    ],
                    "initialization": "random",
                },
            }
        }
        tl_params = recursive_dict_update(tl_params_default, tl_params)

        vehicles_params = {
            "min_gap": 2.5,
            "max_speed": 30,
            "decel": 7.5,
            "speed_mode": "right_of_way",
        }

        if sim_params is None:
            sim_params = {
                "restart_instance": True,
                "sim_step": 1,
                "print_warnings": False,
                "render": False,
            }

        env_params = {
            "horizon": horizon,
            "additional_params": {
                "target_velocity": 50,
                "switch_time": 3,
                "num_observed": 2,
                "discrete": False,
                "tl_type": "actuated",
                "num_local_edges": 4,
                "num_local_lights": 4,
            },
        }

        net_params = {
            "additional_params": {
                "speed_limit": 35,  # inherited from grid0 benchmark
                "grid_array": {
                    "row_inner_lengths": row_inner_lengths,
                    "col_inner_lengths": col_inner_lengths,
                    "short_length": short_length,
                    "long_length": long_length,
                    "cars_left": 0,
                    "cars_right": 0,
                    "cars_top": 0,
                    "cars_bot": 0,
                },
                "horizontal_lanes": 1,
                "vertical_lanes": 1,
            }
        }

        initial_config_params = {"spacing": 'custom', "shuffle": True}
        env_params["additional_params"] = {}

        GenericGridEnv.__init__(
            self=self,
            tl_params=tl_params,
            agents_params=agents_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            env_state_params=env_state_params,
            inflow_type=inflow_type,
            inflow_params=inflow_params,
            simulator=simulator,
            vehicles_params=vehicles_params,
            env_params=env_params,
            sim_params=sim_params,
            net_params=net_params,
            initial_config_params=initial_config_params,
            detector_params=detector_params,
            action_repeat_params=action_repeat_params
        )
