import json
import os
from pathlib import Path

from ray.tune import register_trainable

from wolf.utils.os import empty_directory, makedirs, load_config_file
from wolf.utils.math import recursive_dict_update
import logging


class Ray(dict):

    def __init__(self, *args, **kw):
        super(Ray, self).__init__(*args, **kw)
        self._LOGGER = logging.getLogger(__name__)

    def init(self):
        if "init" in self:
            return self["init"]
        else:
            self._LOGGER.warning("No ray init parameters in configuration file")
            return {}

    def run_params(self):
        if "run" in self:
            return self["run"]
        else:
            self._LOGGER.warning("No ray run parameters in configuration file")
            return {}

    def config(self):
        return self["config"]


class Configuration(object):

    def __init__(self, *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        import logging
        self.logger = logging.getLogger(__name__)
        self.dict = {}
        self.device = None
        self.plt = None
        self.backend = None
        self.R = None

    def __getitem__(self, arg):
        self.__check__()
        return self.dict[arg]

    def has_key(self, key):
        return key in self.dict

    def __contains__(self, key):
        return key in self.dict

    def __str__(self):
        import pprint
        return pprint.pformat(self.dict)

    def __check__(self):
        if not self.dict:
            raise Exception("please load the configuration file")

    def ray(self):
        return self._ray

    def create_fresh_workspace(self, force=False):
        r = ''
        while r != 'y' and r != 'n':
            if force:
                r = 'y'
            else:
                r = input("are you sure you want to erase workspace {} [y/n] ?".format(self.workspace))
            from wolf.utils.os import makedirs
            if r == 'y':
                self.__check__()
                self.__clean_workspace()
                makedirs(self.workspace)
            elif r == 'n':
                makedirs(self.workspace)
            else:
                print("Only [y/n]")
        return self

    def load_sumo(self):
        import os, sys
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            if "sumo_home" in self["general"]:
                os.environ['SUMO_HOME'] = str(self["general"]["sumo_home"])
                os.environ['PATH'] = str(self["general"]["sumo_home"]) + ":" + os.environ["PATH"]
            else:
                sys.exit("please declare environment variable 'SUMO_HOME' or set it in config file")
        return self

    def __clean_workspace(self):
        self.__check__()
        import os
        os.system("rm -rf {}".format(self.workspace))
        return self

    def set_seed(self, seed):
        if seed is not None:
            self.logger.info("Setting seed = {}".format(seed))
            import random
            random.seed(seed)
            import numpy as np
            np.random.seed(seed)
            import torch
            if seed is not None:  # pytorch doesn't like None seed
                torch.manual_seed(seed)
            # TODO: applying seed to tensorflow causes error in tdtse_models.
            # import tensorflow
            # self.logger.info("tensorflow version={}".format(tensorflow.__version__))
            # if int(tensorflow.__version__[0]) >= 2:
            #     tensorflow.random.set_seed(seed)
            # else:
            #     tensorflow.random.set_random_seed(seed)
            try:
                # setting simulation seed
                _ = self["traffic_params"]["flow_params"]["sim_params"]
                self["traffic_params"]["flow_params"]["sim_params"]["seed"] = seed
            except Exception:
                logging.warning("Seed {} hasn't been set to Flow.".format(seed))

            try:
                # setting ray seed
                exps = self["ray"]["run_experiments"]["experiments"]
                for exp, value in exps.items():
                    value["config"]["seed"] = seed
            except Exception:
                logging.warning("Seed {} hasn't been set for Ray.".format(seed))

    def load_custom_models(self):
        from ray.rllib.models import ModelCatalog
        from wolf.world.environments.wolfenv.models.tdtse_models import TdtseCnnTfModel
        ModelCatalog.register_custom_model("tdtse", TdtseCnnTfModel)
        from wolf.world.environments.wolfenv.models.tdtse_models import TdtseCnnTfModelSoheil
        ModelCatalog.register_custom_model("tdtse_soheil", TdtseCnnTfModelSoheil)
        from wolf.world.environments.wolfenv.models.tdtse_models import TdtseCnnTorchModel
        ModelCatalog.register_custom_model("tdtse_torch", TdtseCnnTorchModel)
        from wolf.world.environments.wolfenv.models.queue_models import QueueObsModel
        ModelCatalog.register_custom_model("queue_obs_model", QueueObsModel)
        
        return self

    def load(self, config, override_config=None):
        self.dict = load_config_file(config)

        if override_config:
            override_config = load_config_file(override_config)
            recursive_dict_update(self.dict, override_config)

        if "general" in self.dict:
            if "logging" in self.dict["general"]:
                import logging.config as log_config
                log_config.dictConfig(self.dict["general"]["logging"])

            self.logger.info("load config file at {}".format(config))

            self.id = self.dict["general"].get("id", "no_id")
            self.workspace = Path(self.dict["general"].get("workspace", ""))

            self.seed = None
            if "seed" in self.dict["general"]:
                self.seed = self.dict["general"]["seed"]
            self.set_seed(self.seed)

            import numpy as np
            np.set_printoptions(precision=2, suppress=True)

            self.writer = None
            if "is_tensorboardX" in self["general"]:
                self.is_tensorboardX = self["general"]["is_tensorboardX"]
            else:
                self.is_tensorboardX = False

        if "ray" in self.dict:
            self._ray = Ray(self["ray"])

        return self

    def load_custom_trainable(self):
        from wolf.utils.policy_evaluator import PolicyEvaluator
        register_trainable("EVALUATOR", PolicyEvaluator)
        return self

    def load_matplotlib(self, backend=None):
        if self.plt is not None:
            self.logger.warning("matplotlib already loaded")
        else:
            self.backend = None
            import matplotlib
            if backend is not None:
                self.backend = backend
            elif self["general"]["matplotlib_backend"] is not None:
                self.backend = self["general"]["matplotlib_backend"]
            if self.backend is not None:
                matplotlib.use(self.backend)
            import matplotlib.pyplot as plt
            self.plt = plt
        return self

    def load_pytorch(self, override_device_str=None):

        self.logger.warning("Using import torch.multiprocessing as multiprocessing")
        self.logger.warning("Using multiprocessing.set_start_method('spawn')")
        import torch.multiprocessing as multiprocessing
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError as e:
            self.logger.warning(str(e))

        if self.device is not None:
            self.logger.warning("pytorch already loaded")
        else:
            if override_device_str is not None:
                import torch
                _device = torch.device(override_device_str)
            else:
                from wolf.utils.torch_utils import get_the_device_with_most_available_memory
                _device = get_the_device_with_most_available_memory()
            self.device = _device
            self.logger.info("DEVICE : {}".format(self.device))

        return self

    def load_tensorboardX(self):
        if self.is_tensorboardX:
            from tensorboardX import SummaryWriter
            empty_directory(self.workspace / "tensorboard")
            makedirs(self.workspace / "tensorboard")
            # exit()

            self.writer = SummaryWriter(str(self.workspace / "tensorboard"))
            command = "tensorboard --logdir {} --port 6009 &".format(str(self.workspace / "tensorboard"))
            self.logger.info("running command \"{}\"".format(command))
            os.system(command)

    def dump_to_workspace(self, filename="config.json"):
        """
        Dump the configuration a json file in the workspace.
        """
        makedirs(self.workspace)
        print(self.dict)
        with open(self.workspace / filename, 'w') as f:
            json.dump(self.dict, f, indent=2)


C = Configuration()
