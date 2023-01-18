from flow.networks import Network
from flow.controllers import SimCarFollowingController, GridRouter
from flow.core.params import (
    VehicleParams,
    SumoCarFollowingParams,
    SumoParams,
    EnvParams,
    NetParams,
    InitialConfig,
    DetectorParams,
    InFlows,
)
from wolf.world.environments.wolfenv.wolf_env import WolfEnv
import pprint
import numpy as np
import logging
import pathlib
import copy


class RealWorldNetworkEnv(WolfEnv):
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
        env_state_params=None,
        action_repeat_params=None,
        simulator='traci',
    ):
        self.logger = logging.getLogger(__name__)

        vehicles = VehicleParams()
        vehicles.add(
            veh_id="human",
            acceleration_controller=(SimCarFollowingController, {}),
            car_following_params=SumoCarFollowingParams(**vehicles_params),
            routing_controller=(GridRouter, {}),
            num_vehicles=0,
        )

        initial_config = InitialConfig(**initial_config_params)

        # handle the relative path
        wolf_path = str(pathlib.Path(__file__).parent.absolute().parents[3]) + '/'
        abs_template_paths = copy.deepcopy(net_params["template"])
        for key in net_params["template"]:
            abs_template_paths[key] = wolf_path + net_params["template"][key]

        net_params_obj = NetParams(template=abs_template_paths)

        network = Network(
            name="real_world_network",
            vehicles=vehicles,
            net_params=net_params_obj,
            initial_config=initial_config,
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
            env_state_params=env_state_params,
            controlled_nodes=net_params["controlled_tls"],
            action_repeat_params=action_repeat_params,
            simulator=simulator,
        )
