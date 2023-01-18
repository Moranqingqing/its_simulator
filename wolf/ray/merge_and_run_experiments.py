import os

import pprint
import argparse

from wolf.ray.main import runs
from pathlib import Path


def run(baseline_paths, workspace="tmp", sumo_home=Path.home() / "sumo_binaries" / "bin"):
    config_file = dict(
        ray=dict(
            local_mode=False,
            log_to_driver=True,
            logging_level="WARNING",
            run_experiments=dict(
                experiments=dict()
            )
        ),
        general=dict(
            id="run_all_baselines",
            seed=None,
            is_tensorboardX=False,
            sumo_home=str(sumo_home),
            workspace=str(workspace)
        )
    )

    config_file["general"]["logging"] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(name)s] %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler"
            }
        },
        "loggers": {
            "": {
                "handlers": [
                    "default"
                ],
                "level": "WARNING",
                "propagate": False
            },
            "some.logger.you.want.to.enable.in.the.code": {
                "handlers": [
                    "default"
                ],
                "level": "ERROR",
                "propagate": False
            }
        }
    }

    for file in baseline_paths:
        _dict = None
        print("reading {}".format(file))
        with open(file, 'r') as infile:
            if "json" in file:
                import json

                _dict = json.load(infile)
            elif "yaml" in file:
                import yaml

                _dict = yaml.full_load(infile)
            else:
                raise Exception("wrong file format")
        key, value = next(iter(_dict["ray"]["run_experiments"]["experiments"].items()))
        if key in config_file["ray"]["run_experiments"]["experiments"]:
            raise Exception("Experiment {} in double".format(key))
        config_file["ray"]["run_experiments"]["experiments"][key] = value
        value["local_dir"] = config_file["general"]["workspace"]
        # subprocess.run(["tensorboard", "--logdir",value["local_dir"]+"/"+key],start_new_session=True,shell=True)
        os.system("tensorboard --logdir " + value["local_dir"] + "/" + key + " &")

    config_file["general"]["repeat"] = 1

    pprint.pprint(config_file)

    runs(config_file)


EXAMPLE_USAGE = """
example usage:
    python merge_and_run_experiments.py --files path/to/file.yaml path/to/file2.yaml --sumo_home /home/user/sumo_binaries/bin --workspace tmp
"""


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Wolf] Merge single experiments file into one main experiments',
        epilog=EXAMPLE_USAGE)

    parser.add_argument('-f', '--files', nargs='+', help='<Required> Set flag', required=True)

    # required input parameters
    parser.add_argument(
        '--sumo_home',
        type=str,
        default=Path.home() / "sumo_binaries" / "bin",
        help='Home of sumo')

    parser.add_argument(
        '--workspace',
        type=str,
        default="tmp",
        help='Where to save files.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    run(args.files, args.workspace, args.sumo_home)
