import ray
from ray.rllib.agents.pg import pg
from ray.tune import register_env

from flow.core.params import EnvParams, VehicleParams, NetParams, SumoParams
from gym.spaces import Discrete

from flow.networks import TrafficLightGridNetwork
from sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector
from wolf.world.environments.traffic.agents.connectors.done.mock_done_connector import MockDoneConnector
from sow45.world.environments.traffic.agents.connectors.observation.mock import MockObservationConnector
from sow45.world.environments.traffic.agents.connectors.reward.mock import MockRewardConnector
from sow45.world.environments.traffic.agents.traffic_agent import TrafficAgent
from sow45.world.environments.traffic.phase import rGrG, GrGr
from sow45.world.environments.traffic.traffic_env import TrafficEnv

import logging

logger = logging.getLogger(__name__)


def create_env(env_config=None):
    net_params = NetParams(
        additional_params={
            'grid_array': {
                'row_num': 3,
                'col_num': 2,
                'inner_length': 500,
                'short_length': 500,
                'long_length': 500,
                'cars_top': 20,
                'cars_bot': 20,
                'cars_left': 20,
                'cars_right': 20,
            },
            'horizontal_lanes': 1,
            'vertical_lanes': 1,
            'speed_limit': {
                'vertical': 35,
                'horizontal': 35
            }
        },
    )
    network = TrafficLightGridNetwork(
        name='grid',
        vehicles=VehicleParams(),
        net_params=net_params
    )
    env = TrafficEnv(EnvParams(), SumoParams(), network, simulator='traci')

    traffic_light_1 = TrafficAgent(
        id="tl_agent_center0",
        action_connector=ExtendChangePhaseConnector(node_id="center0", phases=[GrGr, rGrG]),
        observation_connector=MockObservationConnector(Discrete(3)),
        reward_connector=MockRewardConnector(Discrete(100)),
        done_connector=MockDoneConnector())

    env.register_agents([traffic_light_1])
    return env


register_env(name="my_env", env_creator=create_env)

any_agent = next(iter(create_env().get_agents().values()))
tl_policy = (None, any_agent.obs_space(), any_agent.action_space(), {})

policies_graph = {
    "tl_policy": tl_policy
}


def policy_mapping(agent_id):
    if "tl" in agent_id:
        return "tl_policy"


ray.init()

trainer = pg.PGTrainer(
    env="my_env",
    config={
        "num_workers": 1,
        "multiagent": {
            "policies": policies_graph,
            "policy_mapping_fn": policy_mapping}})
#
while True:
    print(trainer.train())
