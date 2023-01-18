# /home/ncarrara/work/sow45_code/wolf/tests/test0/results/global_agent/test/checkpoint_1/checkpoint-1
# --run APEX --env test0 --steps 1000 --video-dir="video_flow"
import os
from argparse import ArgumentParser
from pathlib import Path

import ray
from ray.rllib.rollout import create_parser
from wolf.ray.rollout import run2
from wolf.utils.configuration.configuration import Configuration


def run(type_env, test, agent, steps, episodes, sumo_home):
    """

    :param type_env: ctm, or traffic_env
    :param test:
    :param agent:
    :param steps:
    :param episodes:
    :param sumo_home:
    :return:
    """
    # read run from config file ..
    args = ArgumentParser().parse_args()
    args.render = True

    args.steps = steps
    args.config = {}
    args.episodes = episodes
    args.monitor = None
    args.use_shelve = False
    args.save_info = False
    args.track_progress = False
    args.out = None
    args.no_render = False
    args.sumo_home = sumo_home

    C = Configuration()

    path_config_file = type_env + "/" + test + "/" + agent + ".yaml"
    C.load(path_config_file)
    rlib_algo = C["ray"]["run_experiments"]["experiments"][agent]["run"]
    args.run = rlib_algo
    agent_folder = type_env + "/" + test + "/" + "results" + "/" + agent
    print("agent folder : ", agent_folder)
    for file in os.listdir(agent_folder):
        print(file)
        if rlib_algo in file:
            max = 0
            for checkpoint in os.listdir(agent_folder + "/" + file):
                if "checkpoint" in checkpoint:
                    number = int(checkpoint.split("_")[1])
                    if number > max:
                        max = number

            args.video_dir = "{}/{}/video".format(agent_folder, file)
            args.env = type_env + "_" + test  # (type_env + "_" if type_env == "ctm" else "") + test
            args.checkpoint = "{}/{}/checkpoint_{}/checkpoint-{}".format(agent_folder, file, max - 1, max - 1)
            run2(args, create_parser())
            ray.shutdown()


run("ctm", "test0_1", "global_agent", 1000, 1, Path.home() / "sumo_binaries" / "bin")
