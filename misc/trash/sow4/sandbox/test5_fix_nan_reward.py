"""Multi-agent traffic light example (single shared policy)."""
import ray
from gym.spaces import Discrete
from ray.rllib.agents.pg import pg
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env

# Experiment parameters
from sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector
from wolf.world.environments.traffic.agents.connectors.done.mock_done_connector import MockDoneConnector
from sow45.world.environments.traffic.agents.connectors.observation.mock import MockObservationConnector
from sow45.world.environments.traffic.agents.connectors.reward.mock import MockRewardConnector
from sow45.world.environments.traffic.agents.traffic_agent import TrafficAgent
from sow45.world.environments.traffic.phase import GrGr, rGrG
from sow45.world.environments.traffic.traffic_env import TrafficEnv

N_ROLLOUTS = 63  # number of rollouts per training iteration
N_CPUS = 63  # number of parallel workers

# Environment parameters
HORIZON = 400  # time horizon of a single rollout
V_ENTER = 30  # enter speed for departing vehicles
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 1, 1, 1, 1

EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
N_ROWS = 3  # number of row of bidirectional lanes
N_COLUMNS = 3  # number of columns of bidirectional lanes

# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
num_vehicles = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=num_vehicles)

# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    inflow.add(
        veh_type="human",
        edge=edge,
        vehs_per_hour=EDGE_INFLOW,
        departLane="free",
        departSpeed=V_ENTER)

# sumo-related parameters (see flow.core.params.SumoParams)
sim_params = SumoParams(
    restart_instance=True,
    sim_step=1,
    render=True,
)

# environment related parameters (see flow.core.params.EnvParams)
env_params = EnvParams(
    horizon=HORIZON,
    additional_params={
        "target_velocity": 50,
        "switch_time": 3,
        "num_observed": 2,
        "discrete": False,
        "tl_type": "actuated",
        "num_local_edges": 4,
        "num_local_lights": 4,
    },
)

# network-related parameters (see flow.core.params.NetParams and the
# network's documentation or ADDITIONAL_NET_PARAMS component)
net_params = NetParams(
    inflows=inflow,
    additional_params={
        "speed_limit": V_ENTER + 5,  # inherited from grid0 benchmark
        "grid_array": {
            "short_length": SHORT_LENGTH,
            "inner_length": INNER_LENGTH,
            "long_length": LONG_LENGTH,
            "row_num": N_ROWS,
            "col_num": N_COLUMNS,
            "cars_left": N_LEFT,
            "cars_right": N_RIGHT,
            "cars_top": N_TOP,
            "cars_bot": N_BOTTOM,
        },
        "horizontal_lanes": 1,
        "vertical_lanes": 1,
    },
)

# parameters specifying the positioning of vehicles upon initialization
# or reset (see flow.core.params.InitialConfig)
initial_config = InitialConfig(
    spacing='custom',
    shuffle=True
)

network = TrafficLightGridNetwork(
    name="taffic_light_grid_network",
    vehicles=vehicles,
    net_params=net_params,
    initial_config=initial_config
)


#################################################
#################################################
#################################################
#################################################


def create_env(*_):
    env = TrafficEnv(env_params=env_params,
                     sim_params=sim_params,
                     network=network,
                     simulator="traci")
    agents = []

    phases = [GrGr, rGrG]

    for node_id in env.network.traffic_lights.get_properties().keys():
        tl = TrafficAgent(
            id="tl_{}".format(node_id),
            action_connector=ExtendChangePhaseConnector(node_id=node_id, phases=phases),
            observation_connector=MockObservationConnector(Discrete(3)),
            reward_connector=MockRewardConnector(Discrete(100)),
            done_connector=MockDoneConnector())
        agents.append(tl)

    env.register_agents(agents)
    return env


env_name = "traffic_env_test_3"

# Register as rllib env
register_env(env_name, create_env)

any_agent = next(iter(create_env().get_agents().values()))  # get any agent as they all share the same act/obs space

tl_policy = (None, any_agent.obs_space(), any_agent.action_space(), {})

policies_graph = {
    "tl_policy": tl_policy
}


def policy_mapping(agent_id):
    if "tl" in agent_id:
        return "tl_policy"


ray.init()

trainer = pg.PGTrainer(
    env=env_name,
    config={
        "num_workers": 1,
        "multiagent": {
            "policies": policies_graph,
            "policy_mapping_fn": policy_mapping}})

while True:
    print(trainer.train())
