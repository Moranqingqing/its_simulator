from abc import abstractmethod, ABC
import logging

LOGGER = logging.getLogger(__name__)


class EnvsCollection:

    def __str__(self):
        return self.env_prefix

    def __init__(self, env_prefix, backends=None):
        self.env_prefix = env_prefix
        self.envs = {}
        if backends is None:
            backends = ['gym', 'rllib']
        self.backends = backends

    def register_env(self, env_creator, config):
        name = self.env_prefix + "-" + str(len(self.envs))
        self.envs[name] = env_creator, config
        from rltl.utils.registry import R
        # if name in R.d["envs"]:
        #     raise Exception("Envs \"{}\"is already registered".format(name))

        R.d["envs"][name] = env_creator, config
        return name


class EnvsDistribution(EnvsCollection):

    def __init__(self, env_prefix, backends=None):
        EnvsCollection.__init__(self, env_prefix, backends)

    @abstractmethod
    def _reconstruct(self, config):
        raise NotImplementedError()

    def reconstruct(self, config, gym_name=None):
        env_creator = self._reconstruct(config)
        name = self.register_env(env_creator, config)
        return name

    @abstractmethod
    def _sample(self):
        raise NotImplementedError()

    def sample(self):
        env_creator, config = self._sample()
        name = self.register_env(env_creator, config)
        return name, config


import numpy as np


class FixedSamplingDistribution(EnvsDistribution):

    def __init__(self, setup_envs, env_prefix, backends=None):
        super().__init__(env_prefix=env_prefix, backends=backends)
        self._envs = setup_envs()

    def _sample(self):
        config, env_creator = list(self._envs.items())[np.random.randint(0, len(self._envs))]
        return env_creator, config

    def _reconstruct(self, env_name):
        return self._envs[env_name], env_name


class SampleKwargsEnvsDistribution(EnvsDistribution):

    def __init__(self, env_prefix, env_cls_or_factory, backends=None):
        super().__init__(env_prefix, backends=backends)
        self.env_cls_of_factory = env_cls_or_factory

    @abstractmethod
    def _sample_kwargs(self):
        """
        :return: a dictionary of kwargs for the env constructor
        """
        raise NotImplementedError

    def _reconstruct(self, config):
        return lambda: self.env_cls_of_factory(**config)

    def _sample(self):
        kkwargs = self._sample_kwargs()
        return lambda local_kkwars=kkwargs: self.env_cls_of_factory(**local_kkwars), kkwargs
