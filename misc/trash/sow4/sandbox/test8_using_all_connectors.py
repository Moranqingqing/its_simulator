"""Multi-agent traffic light example (single shared policy)."""
import ray
from ray.rllib.agents.pg import pg
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.core.params import DetectorParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env

# Experiment parameters
from wolf.world.environments.traffic.models.tdtse_models import TdtseCnnTfModel
from sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector
from wolf.world.environments.traffic.agents.connectors.done.mock_done_connector import MockDoneConnector
from sow45.world.environments.traffic.agents.connectors.observation.dtse import DTSEConnector
from sow45.world.environments.traffic.agents.connectors.reward.queue_reward_connector import QueueRewardConnector

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
    print_warnings=False
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

detector_params = DetectorParams()
for i in range(9):
    detector_params.add_induction_loop_detectors_to_intersection(name=f"det_{i}", node_id=f"center{i}",
                                                                 positions=[-5, -100], frequency=100)

network = TrafficLightGridNetwork(
    name="taffic_light_grid_network",
    vehicles=vehicles,
    net_params=net_params,
    initial_config=initial_config,
    detector_params=detector_params
)

traffic_light_params = {
    "green_min": 10,
    "green_max": 60,
    "yellow_max": 3,
    "red_max": 2
}

observation_params = {
    "num_history": 60,
    "detector_position": [5, 100],
    "num_cell": 40,
    "cell_length": 5
}


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

    kernel = env.get_kernel()

    for node_id in env.network.traffic_lights.get_properties().keys():
        tl = TrafficAgent(
            id="tl_{}".format(node_id),
            action_connector=ExtendChangePhaseConnector(node_id=node_id,
                                                        phases=phases,
                                                        tl_params=traffic_light_params,
                                                        kernel=kernel),
            # observation_connector=TDTSEConnector(node_id=node_id,
            #                                      tl_logic=phases,
            #                                      obs_params=observation_params,
            #                                      phase_channel=True,
            #                                      kernel=kernel),
            observation_connector=DTSEConnector(node_id=node_id,
                                                max_speed=V_ENTER+5,
                                                obs_params=observation_params,
                                                kernel=kernel),
            reward_connector=QueueRewardConnector(node_id=node_id, n=100, kernel=kernel),
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

###################################################
################ custome model ####################
###################################################
tf = try_import_tf()

class MyKerasModel(TFModelV2):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(TdtseCnnTfModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        # Define the core model layers which will be used by the other
        # output heads of DistributionalQModel
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")

        conv_1 = tf.keras.layers.Conv2D(32, [1, 8], activation=tf.nn.elu)(self.inputs)
        conv_2 = tf.keras.layers.Conv2D(32, [1, 4], activation=tf.nn.elu)(conv_1)
        conv_3 = tf.keras.layers.Conv2D(32, [1, 2], activation=tf.nn.elu)(conv_2)
        flat = tf.keras.layers.Flatten()(conv_3)
        
        layer_1 = tf.keras.layers.Dense(128, name="my_layer1", activation=tf.nn.relu, kernel_initializer=normc_initializer(1.0))(flat)
        layer_out = tf.keras.layers.Dense(num_outputs, name="my_out", activation=None, kernel_initializer=normc_initializer(1.0))(layer_1)
        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
        self.register_variables(self.base_model.variables)

    # Implement the core forward method
    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return self._value_out

    def metrics(self):
        return {"foo": tf.constant(42.0)}


ModelCatalog.register_custom_model("keras_model", MyKerasModel)

ray.init(local_mode=True)

trainer = pg.PGTrainer(
    env=env_name,
    config={
        "num_workers": 1,
        "multiagent": {
            "policies": policies_graph,
            "policy_mapping_fn": policy_mapping
        },
        "model": {
            "custom_model": "keras_model"
        }
    }
)

for i in range(5000):
    result = trainer.train()
    print(result)

    if i % 10 == 0:
        checkpoint = trainer.save()
        print("############################# checkpoint saved at: {} ############################", checkpoint)
    
    print('')
