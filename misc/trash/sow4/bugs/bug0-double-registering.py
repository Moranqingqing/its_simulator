import gym
from ray.rllib.agents.ppo import ppo
from sympy import pretty_print
from sow45.utils.configuration.configuration import Configuration
from sow45.world.environments.factories.env_meta_factory import META_FACTORY
import ray
import logging


C = Configuration().load("configs/debug.json")

LOGGER = logging.getLogger(__name__)

# gym env factory
create_env, gym_name = META_FACTORY.create_factory(**C["traffic_params"])

# get any agent as they all share the same act/obs space
# test_env = gym.make(gym_name) # TODO pop zero registration
test_env = create_env() # TODO pop double registration
any_agent = next(iter(test_env.get_agents().values()))

# policies, polices can be shared by several agents
policy_graphs = {
    "tl_policy": (None, any_agent.obs_space(), any_agent.action_space(), {})
}

# maps agent ids to policies
def policy_mapping_fn(agent_id):
    if "tl" in agent_id:
        return "tl_policy"

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config["eager"] = False
config["multiagent"] = {
    "policies": policy_graphs,
    "policy_mapping_fn": policy_mapping_fn
}

# run RLlibs experiments
ray.init()

config = ppo.DEFAULT_CONFIG.copy()
trainer = ppo.PPOTrainer(config=config, env=gym_name)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)

# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Trainer's Policy's ModelV2
# (tf or torch) by doing:
trainer.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch trainers/models.


