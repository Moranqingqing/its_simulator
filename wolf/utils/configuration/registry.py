import pprint

from wolf.world.environments.wolfenv.agents.agent_factory import AllTheSameTrafficLights, GlobalTrafficLightsAgent, AllTheSameVehicles
from wolf.world.environments.wolfenv.agents.connectors.action.mock import MockActionConnector
from wolf.world.environments.wolfenv.agents.connectors.action.cycle import CycleConnector
from wolf.world.environments.wolfenv.agents.connectors.action.fixed_order_cycle import FixedOrderCycleConnector
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector
from wolf.world.environments.wolfenv.agents.connectors.action.variable_phasing import VariablePhasingConnector
from wolf.world.environments.wolfenv.agents.connectors.action.phase_select import PhaseSelectConnector
from wolf.world.environments.wolfenv.agents.connectors.action.veh_act_connector import VehActionConnector
from wolf.world.environments.wolfenv.agents.connectors.action.veh_act_connector import VehActionConnector_lc

from wolf.world.environments.wolfenv.agents.connectors.observation.tdtse import TDTSEConnector
from wolf.world.environments.wolfenv.agents.connectors.observation.mock import MockObservationConnector
from wolf.world.environments.wolfenv.agents.connectors.observation.queue_obs_connector import QueueObservationConnector
from wolf.world.environments.wolfenv.agents.connectors.observation.dtse import DTSEConnector
from wolf.world.environments.wolfenv.agents.connectors.observation.veh_obs_connector import CarFollowingConnector, BCMObsConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.cumulative_delay_reward_connector import \
    DifferenceInCumulativeDelayRewardConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.queue_reward_connector import QueueRewardConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.travel_time_reward_connector import \
    TravelTimeRewardConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.average_delay_measurement_connector import \
    AverageDelayMeasurementConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.average_travel_time_measurement_connector import \
    AverageTravelTimeMeasurementConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.approximated_travel_time_reward_connector import \
    ApproximatedTravelTimeRewardConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.approximated_vehicle_count_reward_connector import \
    ApproximatedVehicleCountRewardConnector
from wolf.world.environments.wolfenv.agents.connectors.reward.veh_reward_connector import VehRewardConnector, BCMVehRewardConnector

from wolf.world.environments.wolfenv.agents.multi_agent_settings_factory import SingleGroup, SharedPolicy, \
    IndependentPolicy
from wolf.world.environments.wolfenv.policies.random_policy import RandomPolicy
from wolf.world.environments.wolfenv.policies.static_policy import StaticMinPolicy, StaticMaxPolicy, \
    GlobalGreenWavePolicy
from wolf.world.environments.env_factories import *
from wolf.world.environments.wolfenv.agents.connectors.reward.travel_time_reward_connector import \
    TravelTimeRewardConnector
from wolf.world.environments.wolfenv.agents.connectors.env_state_connector import MockEnvState, \
    AllAgentObservationsEnvState, ConcatSimilarAgentsBoxes
from wolf.world.environments.wolfenv.kernels.tl_wolf_kernel import SecondBasedTrafficLight, \
    CycleBasedTrafficLight, FixedOrderCycleBasedTrafficLight, PhaseSplitTrafficLight


class Registry:
    def __init__(self):
        self.registery = {
            "connectors_classes": {},
            "agent_factories": {},
            "env_factories": {},
            "policies_classes": {},
            "traffic_light_classes": {},
            "true_state_classes": {},
            "group_agents_params_factories": {},
            "multi_agent_config_factories": {},
            "inflow_schedules": {}
        }

    def policy_class(self, name):
        return self.registery["policies_classes"][name]

    def register_policy(self, name, cls):
        self.registery["policies_classes"][name] = cls

    def register_connector(self, name, cls):
        self.registery["connectors_classes"][name] = cls

    def register_agent_factory(self, name, object):
        self.registery["agent_factories"][name] = object

    def register_true_state_class(self, name, cls):
        self.registery["true_state_classes"][name] = cls

    def env_factory(self, env_name):
        if env_name not in self.registery["env_factories"]:
            print("[WARNING] Env {} is not registered, attempting to parse it to create a benchmark env.")
            return lambda config: benchmark_env(env_config=config, **parse_benchmark_params(env_name))
        return self.registery["env_factories"][env_name]

    def register_traffic_light(self, name, cls):
        self.registery["traffic_light_classes"][name] = cls

    def traffic_light_class(self, name):
        return self.registery["traffic_light_classes"][name]

    def connector_class(self, name):
        return self.registery["connectors_classes"][name]

    def agent_factory(self, name):
        return self.registery["agent_factories"][name]

    def true_state_class(self, name):
        return self.registery["true_state_classes"][name]

    def register_env_factory(self, name, factory):
        from ray.tune.registry import register_env
        register_env(name, factory)
        self.registery["env_factories"][name] = factory

    def register_group_agents_params_factory(self, name, object):
        self.registery["group_agents_params_factories"][name] = object

    def group_agents_params_factory(self, name):
        return self.registery["group_agents_params_factories"][name]

    def register_multi_agent_config_factory(self, name, object):
        self.registery["multi_agent_config_factories"][name] = object

    def multi_agent_config_factory(self, name):
        return self.registery["multi_agent_config_factories"][name]

    def register_inflow_schedule(self, name, cls):
        self.registery["inflow_schedules"][name] = cls

    def inflow_schedule(self, name):
        return self.registery["inflow_schedules"][name]

    def __str__(self):
        return pprint.pformat(self.registery)


R = Registry()

# observation connectors
# R.register_connector("MockObservationConnector", MockObservationConnector)
R.register_connector("TDTSEConnector", TDTSEConnector)
R.register_connector("DTSEConnector", DTSEConnector)
R.register_connector("QueueObservationConnector", QueueObservationConnector)
R.register_connector("CarFollowingConnector", CarFollowingConnector)
R.register_connector("BCMObsConnector", BCMObsConnector)
# action connectors
# R.register_connector("MockActionConnector", MockActionConnector)
R.register_connector("ExtendChangePhaseConnector", ExtendChangePhaseConnector)
R.register_connector("PhaseSelectConnector", PhaseSelectConnector)
R.register_connector("VariablePhasingConnector", VariablePhasingConnector)
R.register_connector("CycleConnector", CycleConnector)
R.register_connector("FixedOrderCycleConnector", FixedOrderCycleConnector)
R.register_connector("VehActionConnector", VehActionConnector)
R.register_connector("VehActionConnector_lc", VehActionConnector_lc)

# reward connectors
R.register_connector("QueueRewardConnector", QueueRewardConnector)
R.register_connector("CumulativeDelayRewardConnector", DifferenceInCumulativeDelayRewardConnector)
R.register_connector("TravelTimeRewardConnector", TravelTimeRewardConnector)
R.register_connector("AverageDelayMeasurementConnector", AverageDelayMeasurementConnector)
R.register_connector("AverageTravelTimeMeasurementConnector", AverageTravelTimeMeasurementConnector)
R.register_connector("ApproximatedTravelTimeRewardConnector", ApproximatedTravelTimeRewardConnector)
R.register_connector("ApproximatedVehicleCountRewardConnector", ApproximatedVehicleCountRewardConnector)
R.register_connector("VehRewardConnector", VehRewardConnector)
R.register_connector("BCMVehRewardConnector", BCMVehRewardConnector)




R.register_agent_factory("all_the_same_traffic_lights_agents", AllTheSameTrafficLights())
R.register_agent_factory("global_traffic_lights_agent", GlobalTrafficLightsAgent())
R.register_agent_factory("all_the_same_vehicles_agents", AllTheSameVehicles())

# for retro compatibility
R.register_agent_factory("all_the_same", R.agent_factory("all_the_same_traffic_lights_agents"))
R.register_agent_factory("global_agent", R.agent_factory("global_traffic_lights_agent"))




R.register_group_agents_params_factory("single_group", SingleGroup())
R.register_multi_agent_config_factory("shared_policy", SharedPolicy())
R.register_multi_agent_config_factory("independent_policy", IndependentPolicy())

R.register_policy("random", RandomPolicy)
R.register_policy("static_min", StaticMinPolicy)
R.register_policy("static_max", StaticMaxPolicy)
R.register_policy("green_wave", GlobalGreenWavePolicy)

R.register_env_factory("simple_grid", simple_grid)
R.register_env_factory("generic_grid", generic_grid)
R.register_env_factory("grid_master_slaves_3", lambda config: grid_master_slaves(config, 3, 300))
R.register_env_factory("grid_gaussian_master_slaves_4", lambda config: grid_gaussian_master_slaves(config, 4, 300))
R.register_env_factory("traffic_env_test0", test0)
R.register_env_factory("traffic_env_test0_1", test0_1)
R.register_env_factory("traffic_env_test0_2", test0_2)
R.register_env_factory("traffic_env_test0_3", test0_3)
R.register_env_factory("traffic_env_test1", test1)
R.register_env_factory("traffic_env_test2", lambda config: grid_master_slaves(config, 4, 300))
R.register_env_factory("traffic_env_test3_1", test3_1)
R.register_env_factory("traffic_env_test3_2", test3_2)
R.register_env_factory("real_world_network", lambda config: real_world_network(env_config=config, horizon=6300))
R.register_env_factory("default_ctm", lambda config: CtmEnv.create_env(CtmEnv, **config))
R.register_env_factory("ctm_test1", lambda config: ctm_test1(config))
R.register_env_factory("ctm_test2", lambda config: ctm_test2(config))
R.register_env_factory("ctm_test0", lambda config: ctm_test0(config))
R.register_env_factory("ctm_test0_1", lambda config: ctm_test0_1(config))
R.register_env_factory("ctm_test3", lambda config: ctm_test3(config))
R.register_env_factory("ctm_test4", lambda config: ctm_test4(config))
R.register_env_factory("ctm_test5", lambda config: ctm_test5(config))
R.register_env_factory("car_following_test0", lambda config: car_following_test(config))
R.register_env_factory("car_following_eval0", lambda config: car_following_eval(config))
R.register_env_factory("car_following_eval1", lambda config: car_following_eval1(config))

R.register_env_factory("benchmark_0", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="platoon",
    demand_symmetry="unique",
    stationary="stationary",
    layout_symmetry=True,
    n=1,
    m=1
))

R.register_env_factory("benchmark_0p", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="poisson",
    demand_symmetry="mixed",
    stationary="stationary",
    layout_symmetry=True,
    n=1,
    m=1
))

R.register_env_factory("benchmark_1", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="bernoulli",
    demand_symmetry="unique",
    stationary="stationary",
    layout_symmetry=True,
    n=1,
    m=1
))

R.register_env_factory("benchmark_2", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="platoon",
    demand_symmetry="unique",
    stationary="stationary",
    layout_symmetry=True,
    n=1,
    m=2
))


R.register_env_factory("benchmark_3", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="platoon",
    demand_symmetry="master_slaves",
    stationary="stationary",
    layout_symmetry=True,
    n=1,
    m=3
))

R.register_env_factory("benchmark_4", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="gaussian",
    demand_symmetry="master_slaves",
    stationary="stationary",
    layout_symmetry=False,
    n=1,
    m=3
))

R.register_env_factory("benchmark_5", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="gaussian",
    demand_symmetry="master_slaves",
    stationary="stationary",
    layout_symmetry=True,
    n=1,
    m=3
))

R.register_env_factory("benchmark_6", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="bernoulli",
    demand_symmetry="master_slaves",
    stationary="stationary",
    layout_symmetry=True,
    n=1,
    m=3
))

R.register_env_factory("benchmark_7", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="fake",
    demand_distribution="gaussian",
    demand_symmetry="master_slaves",
    stationary="stationary",
    layout_symmetry=True,
    n=3,
    m=3
))


R.register_env_factory("benchmark_8", lambda config: benchmark_env(
    env_config=config,
    real_or_fake="real"
))

# register_env("ctm_test5-v0", lambda config: ctm_test5(config))
# gym.register(
#             id="ctm_test5-v0",
#             kwargs={},
#             entry_point=lambda **kwargs: env_creator(),
#             reward_threshold=None,
#             nondeterministic=False,
#             max_episode_steps=None,
#         )

R.register_traffic_light("second_based", SecondBasedTrafficLight)
R.register_traffic_light("cycle_based", CycleBasedTrafficLight)
R.register_traffic_light("fixed_order_cycle_based", FixedOrderCycleBasedTrafficLight)
R.register_traffic_light("phase_split", PhaseSplitTrafficLight)

R.register_true_state_class("mock", MockEnvState)
R.register_true_state_class("concatenate_agents_observations", AllAgentObservationsEnvState)
R.register_true_state_class("concatenate_agents_boxes", ConcatSimilarAgentsBoxes)
