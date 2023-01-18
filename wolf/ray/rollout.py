from pathlib import Path
from pprint import pprint

from ray.rllib.rollout import create_parser, run
from ray.rllib.utils import deprecation_warning
from ray.tune import register_env

import os

from wolf.utils.configuration.configuration import Configuration
from wolf.world.environments.ctm.ctm_env import CtmEnv


def run2(args, parser):
    if args.sumo_home and not args.sumo_added:
        os.environ['SUMO_HOME'] = str(args.sumo_home)
        os.environ['PATH'] = str(args.sumo_home) + ":" + os.environ["PATH"]

    from wolf.utils.configuration.registry import R
    from wolf.utils.os import makedirs
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from flow.utils.rllib import get_rllib_config
    import pathlib

    C = Configuration()
    C.load_custom_models()
    C.load_custom_trainable()
    original_env_name = args.env
    args.env = args.env + "-v0"

    path = pathlib.Path(args.checkpoint)
    p = path.parent.parent
    config = get_rllib_config(str(p))["env_config"]
    pprint(config)
    test_env = R.env_factory(original_env_name)(config)
    video_dir = None
    if isinstance(test_env, WolfEnv) or isinstance(test_env, CtmEnv):
        # we do not want rllib to wrap a monitor  since those are not gym environments
        video_dir = args.video_dir
        args.video_dir = None

    def create_env(config):
        test_env = R.env_factory(original_env_name)(config)
        if isinstance(test_env, WolfEnv):
            sim_params = config["sim_params"]
            sim_params["print_warnings"] = False
            sim_params["restart_instance"] = True
            sim_params["num_clients"] = 1
            sim_params["save_render"] = True if video_dir else False
            sim_params["render"] = not args.no_render
        elif isinstance(test_env, CtmEnv):
            config["video_dir"] = video_dir
            config["render_steps"] = not args.no_render
        env = R.env_factory(original_env_name)(config)
        if isinstance(env, WolfEnv):
            if env.sim_params.save_render:
                env.path = video_dir
                makedirs(env.path)
        return env

    register_env(args.env, create_env)

    run(args, parser)


if __name__ == "__main__":

    parser = create_parser()
    parser.add_argument(
        '--sumo_home',
        type=str,
        default=Path.home() / "sumo_binaries" / "bin",
        help='home of sumo')

    args = parser.parse_args()
    
    # set the e-greedy configs for evaluation
    if "exploration_config" not in args.config:
        args.config["exploration_config"] = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.0,
            "final_epsilon": 0.0,
            "epsilon_timesteps": 0
        }

    # set the default value of `explore`
    if "explore" not in args.config:
        args.config["explore"] = False

    # qmix and rllib folder names have extra backslashes
    args.checkpoint = args.checkpoint.replace('\\', '')
    print('args.checkpoint:', args.checkpoint)

    if 'SUMO_HOME' in os.environ:
        import sys
        args.sumo_home = os.environ['SUMO_HOME']
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        args.sumo_added = True
    else:
        args.sumo_added = False

    # Old option: monitor, use video-dir instead.
    if args.monitor:
        deprecation_warning("--monitor", "--video-dir=[some dir]")
    # User tries to record videos, but no-render is set: Error.
    if (args.monitor or args.video_dir) and args.no_render:
        raise ValueError(
            "You have --no-render set, but are trying to record rollout videos"
            " (via options --video-dir/--monitor)! "
            "Either unset --no-render or do not use --video-dir/--monitor.")
    # --use_shelve w/o --out option.
    if args.use_shelve and not args.out:
        raise ValueError(
            "If you set --use-shelve, you must provide an output file via "
            "--out as well!")
    # --track-progress w/o --out option.
    if args.track_progress and not args.out:
        raise ValueError(
            "If you set --track-progress, you must provide an output file via "
            "--out as well!")

    run2(args, parser)
