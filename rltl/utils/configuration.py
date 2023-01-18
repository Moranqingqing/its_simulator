import multiprocessing
import os
from pathlib import Path

from rltl.utils.utils_os import set_seed, load_object, override_dictionary, empty_directory, makedirs


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
        self.show_plots = False

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

    def __clean_workspace(self):
        self.__check__()
        import os
        os.system("rm -rf {}".format(self.workspace))
        return self

    # def load_custom_models(self):
    #     from ray.rllib.models import ModelCatalog
    #     from wolf.world.environments.traffic.models.tdtse_models import TdtseCnnTfModel
    #     ModelCatalog.register_custom_model("tdtse", TdtseCnnTfModel)
    #     ModelCatalog.register_custom_model("tdtse_torch", TdtseCnnTorchModel)
    #     from wolf.world.environments.traffic.models.queue_models import QueueObsModel
    #     ModelCatalog.register_custom_model("queue_obs_model", QueueObsModel)
    #
    #     return self

    def load_tensorflow(self):
        # if "use_tensorflow_rocm" in self.dict["general"] and self.dict["general"]["use_tensorflow_rocm"]:
        #     os.environ["LD_LIBRARY_PATH"] = "/opt/rocm/hip/lib"
        # multiprocessing.set_start_method('spawn')
        import tensorflow as tf
        self.gpus = []
        if "use_gpu" in self.dict["general"] and self.dict['general']['use_gpu']:
            if self.multiprocessing:
                raise Exception("TODO, make GPU and multiproc compatible, using spawn method")
            self.gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(len(self.gpus)-1)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'



        return self


    def load(self, config, override_config=None):
        # print("loading dict !!!! {}".format(len(config)))
        self.dict = load_object(config)

        if override_config:
            override_config = load_object(override_config)
            override_dictionary(self.dict, override_config)
        # print(self.dict)
        if "general" in self.dict:
            if "logging" in self.dict["general"]:
                import logging.config as log_config
                log_config.dictConfig(self.dict["general"]["logging"])

            self.logger.info("load config file at {}".format(config))

            self.id = self.dict["general"].get("id", "no_id")
            self.workspace = Path(self.dict["general"].get("workspace", ""))
            # print(str(self.workspace)+ "!!!!!!!!!!!!!!!!!!!!!!!!")
            if "multiprocessing" in self.dict["general"]:
                self.multiprocessing = self.dict["general"]["multiprocessing"]
            else:
                self.multiprocessing = False
            self.path = self.workspace / self.id
            self.seed = None
            if "seed" in self.dict["general"]:
                self.seed = self.dict["general"]["seed"]
            set_seed(self.seed, backends=["random", "numpy", "pytorch", "tensorflow"])
            # print("seeed set !!!!")
            import numpy as np
            np.set_printoptions(precision=2, suppress=True)

            self.writer = None
            if "is_tensorboardX" in self["general"]:
                self.is_tensorboardX = self["general"]["is_tensorboardX"]
            else:
                self.is_tensorboardX = False
        # print("done !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return self

    # def load_custom_trainable(self):
    #     from wolf.utils.policy_evaluator import PolicyEvaluator
    #     register_trainable("EVALUATOR", PolicyEvaluator)
    #     return self

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
        return self
        # self.logger.warning("Using import torch.multiprocessing as multiprocessing")
        # self.logger.warning("Using multiprocessing.set_start_method('spawn')")
        # import torch.multiprocessing as multiprocessing
        # try:
        #     multiprocessing.set_start_method('spawn')
        # except RuntimeError as e:
        #     self.logger.warning(str(e))
        #
        # if self.device is not None:
        #     self.logger.warning("pytorch already loaded")
        # else:
        #     if override_device_str is not None:
        #         import torch
        #         _device = torch.device(override_device_str)
        #     else:
        #         from wolf.utils.torch_utils import get_the_device_with_most_available_memory
        #         _device = get_the_device_with_most_available_memory()
        #     self.device = _device
        #     self.logger.info("DEVICE : {}".format(self.device))
        #
        # return self

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


# C: Configuration = Configuration()
