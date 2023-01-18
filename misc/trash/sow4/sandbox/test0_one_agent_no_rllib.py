from flow.core.params import EnvParams, VehicleParams, NetParams, SumoParams
from gym.spaces import Discrete
from flow.networks import TrafficLightGridNetwork
from sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector, \
    ExtendChangeActionSpace
from wolf.world.environments.traffic.agents.connectors.done.mock_done_connector import MockDoneConnector
from sow45.world.environments.traffic.agents.connectors.observation.mock import MockObservationConnector
from sow45.world.environments.traffic.agents.connectors.reward.mock import MockRewardConnector
from sow45.world.environments.traffic.agents.traffic_agent import TrafficAgent
from sow45.world.environments.traffic.phase import GrGr, rGrG
from sow45.world.environments.traffic.traffic_env import TrafficEnv
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

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

logger.info("traffic lights on the network : {}".format(network.traffic_lights.get_properties()))
logger.info("vehicles on the network : {}".format(network.vehicles))

sim_params = SumoParams(
    sim_step=1,
    render=True)

env = TrafficEnv(
    env_params=EnvParams(),
    sim_params=sim_params,
    network=network,
    simulator='traci')

agents = []
for node_id in network.traffic_lights.get_properties().keys():
    tl = TrafficAgent(
        id="tl_{}".format(node_id),
        action_connector=ExtendChangePhaseConnector(node_id=node_id,phases=[GrGr,rGrG]),
        observation_connector=MockObservationConnector(Discrete(3)),
        reward_connector=MockRewardConnector(Discrete(100)),
        done_connector=MockDoneConnector())
    agents.append(tl)

env.register_agents(agents)
env.reset()
for i in range(10000):
    logger.info("step {}".format(i))
    env.step({agent.get_id(): ExtendChangeActionSpace.EXTEND for agent in agents})
logger.info("done")
