import ray
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray.tune import register_trainable, registry, Trainable
from ray.tune.registry import ENV_CREATOR
from ray.tune.resources import Resources

"""
Evaluate a policy. A default_policy must be set in the multiagent policy_graphs. Cf registery for examples of policies.
"""
class PolicyEvaluator(Trainable):
    def __init__(self,config=None, logger_creator=None):
        super().__init__(config, logger_creator)

    def _setup(self, config):
        env_creator = registry._global_registry.get(ENV_CREATOR, config["env"])
        policies = config["multiagent"]["policies"]
        policy_mapping_fn = config["multiagent"]["policy_mapping_fn"]
        rollout_fragment_length = config["timesteps_per_iteration"] / config["num_workers"]
        env_config = config["env_config"]
        policy_config=config["policy_config"] if "policy_config" in config\
            else {"create_env_on_driver": True}
        self.workers = [
            RolloutWorker.as_remote().remote(
                policy=policies,
                policy_config=policy_config,
                env_creator=env_creator,
                rollout_fragment_length=rollout_fragment_length,
                env_config=env_config,
                num_workers=config["num_workers"],
                policy_mapping_fn=policy_mapping_fn)
            for _ in range(config["num_workers"])
        ]

    @classmethod
    def default_resource_request(cls, config):
        return Resources(
            cpu=0,
            gpu=0,
            extra_cpu=config["num_workers"],
            extra_gpu=config["num_gpus"]) #* config["num_workers"])

    def _train(self):
        ray.get([w.sample.remote() for w in self.workers])
        return collect_metrics(remote_workers=self.workers)

    def _save(self, tmp_checkpoint_dir):
        pass

    def _restore(self, checkpoint):
        pass



