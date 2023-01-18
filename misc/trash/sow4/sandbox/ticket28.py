from sow45.utils.configuration.configuration import Configuration
from sow45.world.environments.factories.env_meta_factory import META_FACTORY
from sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector
from wolf.world.environments.traffic.agents.connectors.done.mock_done_connector import MockDoneConnector
from sow45.world.environments.traffic.agents.connectors.observation.mock import MockObservationConnector
from sow45.world.environments.traffic.agents.connectors.reward.mock import MockRewardConnector
from sow45.world.environments.traffic.agents.traffic_agent import TrafficAgent
from sow45.world.environments.traffic.phase import GrGr, rGrG

from gym.spaces import Discrete
import logging
import sys

if len(sys.argv) < 2:
    config_file = "../main/configs/xiaoyu.json"
else:
    config_file = sys.argv[1]  # TODO it is a bit dirty, should use argparse

C = Configuration().load(config_file).create_fresh_workspace(force=True)

LOGGER = logging.getLogger(__name__)
LOGGER.info("[SIMPLE MAIN] config_file={}".format(config_file))
# exit()

# gym env factory
create_env, gym_name = META_FACTORY.create_factory(**C["traffic_params"])

# get any agent as they all share the same act/obs space
test_env = create_env()
any_agent = next(iter(test_env.get_agents().values()))

# policies, polices can be shared by several agents
policy_graphs = {
    "tl_policy": (None, any_agent.obs_space(), any_agent.action_space(), {})
}


# maps agent ids to policies
def policy_mapping_fn(agent_id):
    if "tl" in agent_id:
        return "tl_policy"


# run RLlibs experiments
ray.init(**C.ray().init())

# setup config
config = C.ray().env_config()
horizon = config["horizon"] or C["traffic_params"]["flow_params"]["env_params"]["horizon"]
config["train_batch_size"] = config["train_batch_size"] or horizon * C.ray()["n_rollouts"]
config["num_workers"] = config["num_workers"] or C.ray().init().num_cpus() - 1
config_extra = {
    "horizon": horizon,
    "multiagent": {
        "policies": policy_graphs,
        "policy_mapping_fn": policy_mapping_fn
    }}
config = {**config, **config_extra}

# setup experiments
experiments = {
    "simple_main": {
        **C.ray().experiments_config(),
        **{
            "run": C.ray().default_algorithm(),
            "env": gym_name,
            "config": config
        }
    }
}

trials = run_experiments(experiments, **C.ray().run_experiments())
