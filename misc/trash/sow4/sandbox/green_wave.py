from wolf.world.environments.traffic.models.tdtse_models import MultiNodesTDTSEObsModel
from sow45.world.environments.factories.env_meta_factory import META_FACTORY, GREEN_WAVE

create_env, gym_name = META_FACTORY.create_factory(GREEN_WAVE)

agent_params = {
    "module": "sow45.world.environments.traffic.agents.agent_factory",
    "class_name": "OneAgentForAllIntersections",
    "params": {
        "action_params": {
            "module": "sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase",
            "class_name": "MultiNodesPhaseConnector",
            "params": {
                "phases": [
                    "GrGr",
                    "rGrG"
                ],
                "tl_params": {
                    "green_min": 10,
                    "green_max": 60,
                    "yellow_max": 3,
                    "red_max": 2
                }
            }
        },
        "obs_params": {
            "module": "sow45.world.environments.traffic.agents.connectors.observation.tdtse",
            "class_name": "MultiNodesTDTSEConnector",

            "params": {
                "tl_logic": ["GrGr", "rGrG"],
                "obs_params": {
                    "num_history": 60,
                    "detector_position": [5, 100],
                    "phase_channel": True}
            }
        },
        "reward_params": {
            "module": "sow45.world.environments.traffic.agents.connectors.reward.queue_reward_connector",
            "class_name": "MultiNodesQueueRewardConnector",
            "params": {
                "stop_speed": 0.3,
                "n": 100
            }
        }}}

sim_params = {
    "restart_instance": True,
    "sim_step": 1,
    "render": False
}
env_params = {
    "additional_params": {
        "speed_limit": 35,  # inherited from grid0 benchmark
        "grid_array": {
            "short_length": 300,
            "inner_length": 300,
            "long_length": 100,
            "cars_left": 1,
            "cars_right": 1,
            "cars_top": 1,
            "cars_bot": 1
        },
        "horizontal_lanes": 1,
        "vertical_lanes": 1}}

config = {
    "n_intersections": 10,
    "sim_params": sim_params,
    "env_params": env_params,
    "agents_params": agent_params
}

test_env = create_env(config)
o = test_env.reset()
any_agent = next(iter(test_env.get_agents().values()))
m = MultiNodesTDTSEObsModel(any_agent.obs_space(), any_agent.action_space(), 1, {}, "hello")
for _ in range(10):
    test_env.step({})
