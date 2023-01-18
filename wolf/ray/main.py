import logging
import ray
import pprint

LOGGER = logging.getLogger(__name__)


def on_episode_step(info):
    episode = info["episode"]
    pass


def setup_run_exp_params(C):
    from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper

    from wolf.ray.misc.vizu_exp_schedule import show_schedule
    def resolve_multi_agent_config(spec):
        if "exploration_config" in spec["config"]:
            if "epsilon_schedule" in spec["config"]["exploration_config"]:
                show_schedule(spec["config"]["exploration_config"]["epsilon_schedule"])
        from wolf.utils.configuration.registry import R
        config = spec["config"]
        create_env = R.env_factory(config["env"])
        config["env_config"]["horizon"] = config['horizon']
        env = create_env(config["env_config"])
        if isinstance(env, _GroupAgentsWrapper):
            return env.env.multi_agent_config
        else:
            return env.multi_agent_config

    # setup config
    run_ex_params = C.ray()["run_experiments"]
    # from wolf.utils.configuration.registry import R
    for name, exp in run_ex_params["experiments"].items():
        config = exp["config"]
        # config["env_config"]["R"] = copy.deepcopy(R)
        config["multiagent"] = ray.tune.sample.sample_from(lambda spec: resolve_multi_agent_config(spec))
        config["callbacks"] = {
            "on_episode_step": on_episode_step,
        }
        exp["local_dir"] = C["general"]["workspace"]

        def trial_name_string(trial):
            name = "{}_{}".format(trial.trainable_name, trial.trial_id)
            return name

        exp["trial_name_creator"] = trial_name_string

        # exp["loggers"] = loggers
    LOGGER.info("[setup_experiments] \n\n{}\n".format(pprint.pformat(run_ex_params)))
    # exit()
    return run_ex_params


def runs(config_file_path, override_config_file_path):
    from wolf.utils.configuration.configuration import Configuration
    C = Configuration() \
        .load(config_file_path, override_config_file_path) \
        .load_custom_trainable().load_sumo() \
        .load_custom_models()

    import ray
    from ray.tune import tune
    import logging
    LOGGER = logging.getLogger(__name__)

    # run RLlibs experiments
    ray.init(**C.ray().init())

    params = setup_run_exp_params(C)

    for run in range(C["general"]["repeat"]):
        # TODO to do multiple runs: use Repeater (https://ray.readthedocs.io/en/latest/tune-searchalg.html#variant-generation-grid-search-random-search)
        # but it is not working as expected, so this do that instead
        LOGGER.info("------------------------------------------------")
        LOGGER.info("------------------------------------------------")
        LOGGER.info("------------------    run {}   -----------------".format(run))
        LOGGER.info("------------------------------------------------")
        LOGGER.info("------------------------------------------------")
        trials = tune.run_experiments(**params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run wolf experiments')
    parser.add_argument("config_file", type=str, default="configs/main.yaml", help='path of configuration file')
    parser.add_argument("override_config_file", nargs='?', type=str, default=None, help='path of overriding configuration file')
    args = parser.parse_args()

    runs(args.config_file, args.override_config_file)
