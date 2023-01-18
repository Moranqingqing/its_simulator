import collections
from pathlib import Path

from rltl.main.gan import SUPER_GAN, CLASSIC, AGGLOMERATED_GAN
from rltl.utils.configuration import Configuration
from rltl.utils.registry import R
from rltl.utils.replay_memory import Memory
from rltl.utils.utils_os import save_object, load_object

import os
import tensorflow as tf


class BaselinesData:
    def __init__(self, path):
        self.baselines = collections.OrderedDict()
        self.path = path

    def new_baseline(self, name):
        baseline = BaselineData(self.path / name)
        if name in self.baselines:
            print(" warning Baseline {} already exist".format(name))
        self.baselines[name] = baseline
        return baseline

    def save(self):
        for name, baseline in self.baselines.items():
            baseline.save()


    def load_baseline(self,baseline_name):
        self.baselines[baseline_name].load()

    def load(self, dont_load_models=False):
        if self.path.exists():
            for baseline in self.path.iterdir():
                self.baselines[baseline.name] = BaselineData(baseline)
                self.baselines[baseline.name].load(dont_load_models)
        else:
            print("{} doesn't exist".format(self.path))


class BaselineData:
    def __init__(self, path):
        self.path = path
        self.models = {}

    def get_model(self, perf, type_gan, type_model, agent, name_env=None):
        if type_gan == CLASSIC:
            if (perf, type_gan, type_model, agent, name_env) in self.models:
                return self.models[(perf, type_gan, type_model, agent, name_env)]
            else:
                return None
        else:
            if (perf, type_gan, type_model, agent, "all_envs") in self.models:
                return self.models[(perf, type_gan, type_model, agent, "all_envs")]
            else:
                return None

    def set_model(self, object, perf, type_gan, type_model, agent, name_env=None):
        if type_gan == CLASSIC:
            self.models[(perf, type_gan, type_model, agent, name_env)] = object
        else:
            self.models[(perf, type_gan, type_model, agent, "all_envs")] = object

    def save(self):
        for name, model in self.models.items():
            if model:
                model.save(str(Path(self.path / "models" / "#".join(name))))

    def load(self, dont_load_models=False):
        if not dont_load_models:
            path = Path(self.path / "models")
            if path.exists():
                for model_path in path.iterdir():
                    model_path = str(model_path)
                    # print()
                    # print()
                    print(str(model_path))
                    # print()
                    # print()
                    
                    model = tf.keras.models.load_model(model_path, compile=False)
                    split = os.path.basename(model_path).split("#")
                    self.set_model(model, *split)

            else:
                print("path {} does not exist".format(path))
        # else:
        #     print("These models won't be loaded: {}".format(str(self.path)))
class EnvData:
    def __init__(self):
        self.memory = None
        self.env_config = None
        self.name = None

    def set_name(self, name):
        self.name = name

    def set_env_config(self, config):
        self.env_config = config

    def set_memory(self, memory):
        self.memory = memory

    def save(self, path):
        self.memory.save(path=path / "samples.pickle")
        params = {
            "name": self.name,
            "env_config": self.env_config
        }
        save_object(params, path / "params.json", indent=4)

    def load(self, path):
        self.memory = Memory.static_load(path / "samples.pickle")
        params = load_object(path / "params.json")
        self.name = params["name"]
        self.env_config = params["env_config"]

    def __str__(self):
        return "<name={} config={}>".format(self.name, self.env_config)


class ExperimentData:

    def __init__(self, path, config):
        self.config = config
        self.envs_source = collections.OrderedDict()
        self.envs_target = collections.OrderedDict()
        self.learn_gans_data = BaselinesData(path / "learn_gans")
        self.learn_classifiers_data = BaselinesData(path / "learn_classifiers")
        self.params = {}
        self.path = path
        self.log_folder = self.path / "logs"
        self.envs_path = self.path / "create_datasets"
        self.config_path = self.path / "config.json"
        self.cc_between_sources = None
        self.cc_targets = None
        self.path_cc_between_sources = self.path / "cc_between_sources.json"
        self.path_cc_targets = self.path / "cc_targets.json"
        self.source_collection, self.target_collection = R.get_envs_collection(config["envs_collection"])()
        self.threshold_sigma = {}

    def add_exp_env_state_source(self, ees):
        self.envs_source[ees.name] = ees

    def add_exp_env_state_target(self, ees):
        self.envs_target[ees.name] = ees

    def save(self):
        for type_env in ["source", "target"]:
            for name, exp_env in getattr(self, "envs_" + type_env).items():
                exp_env.save(self.envs_path / type_env / name)
        save_object(self.config.dict, self.config_path, indent=4)
        if self.cc_between_sources:
            save_object(self.cc_between_sources,
                        path=self.path_cc_between_sources,
                        indent=4)
        if self.cc_targets:
            save_object(self.cc_targets,
                        path=self.path_cc_targets,
                        indent=4)

    def load(self, dont_load_models=False):
        for type_env in ["source", "target"]:
            path_envs = self.envs_path / type_env
            if path_envs.exists():
                files = sorted(os.listdir(str(path_envs)))
                for f in files:
                    ees = EnvData()
                    ees.load(path_envs / f)
                    if type_env == "source":
                        self.add_exp_env_state_source(ees)
                    elif type_env == "target":
                        self.add_exp_env_state_target(ees)
                    else:
                        raise Exception()
        self.config = Configuration()

        if self.path_cc_between_sources.exists():
            self.cc_between_sources = load_object(path=self.path_cc_between_sources)

        if self.path_cc_targets.exists():
            self.cc_targets = load_object(path=self.path_cc_targets)

        self.learn_gans_data.load(dont_load_models)
        self.learn_classifiers_data.load(dont_load_models)

        self.threshold_sigma = load_object(self.path / "threshold_context_sigma"/"data.json")
        return self

    def __str__(self):
        x = "Experiment\n\t{}".format(self.config["general"]["id"]) + "\n"
        x += "Params:\n\t" + str(self.params) + "\n"
        x += "Envs:\n\t" + "\n\t".join(["{}={}".format(k, v) for k, v in self.envs_source.items()])
        # print(x)
        return x
