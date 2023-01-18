import argparse

from ray.rllib.agents.pg import pg

from sow45.sow4.main.utils import setup_run_exp_params
from sow45.utils.configuration.configuration import Configuration
from sow45.world.environments.factories.env_meta_factory import META_FACTORY
from sow45.world.environments.traffic.agents.connectors.action.exchange_change_phase import ExtendChangePhaseConnector
from wolf.world.environments.traffic.agents.connectors.done.mock_done_connector import MockDoneConnector
from sow45.world.environments.traffic.agents.connectors.observation.mock import MockObservationConnector
from sow45.world.environments.traffic.agents.connectors.reward.mock import MockRewardConnector

from sow45.world.environments.traffic.agents.traffic_agent import TrafficAgent
from sow45.world.environments.traffic.phase import GrGr, rGrG
import ray
from ray.tune.registry import register_env, _global_registry, TRAINABLE_CLASS
from ray.tune import run_experiments, tune
from gym.spaces import Discrete
import logging

LOGGER = logging.getLogger(__name__)

C = Configuration().load("configs/grid_search.json").create_fresh_workspace(force=True)

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

ray.init(**C.ray().init())

params = setup_run_exp_params(gym_name, policy_graphs, policy_mapping_fn, C)

# trials = tune.run(**params)
trials = tune.run_experiments(**params)

