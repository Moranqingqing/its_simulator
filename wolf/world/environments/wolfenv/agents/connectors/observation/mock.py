from gym.spaces import Discrete

from wolf.world.environments.wolfenv.agents.connectors.observation.obs_connector import ObsConnector

class MockObservationConnector(ObsConnector):

    def __init__(self, **kwargs):
        super().__init__(observation_space=Discrete(1), kernel=None,**kwargs)

    def a_compute(self):
        return self._observation_space.sample()


