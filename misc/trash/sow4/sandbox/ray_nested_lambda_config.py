import ray
from ray import tune

def dummy_fn(config, reporter):
  print(config)

def resolve_b(spec):
  values = [i ** spec.env_config.a for i in range(2, 4)]
  return tune.grid_search(values)

exp_config = {
  "dummy_exp": {
    "run": dummy_fn,
    "config": {"a": tune.grid_search([1, 2]),
               "b": resolve_b},
  },
}

ray.init()
tune.run_experiments(exp_config)