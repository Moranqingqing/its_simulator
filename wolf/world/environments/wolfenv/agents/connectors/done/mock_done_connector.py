from wolf.world.environments.wolfenv.agents.connectors.done.done_connector import DoneConnector
import numpy as np


class MockDoneConnector(DoneConnector):

    def __init__(self):
        super().__init__(kernel=None)

    def compute(self):
        return np.random.random() > 0.75
