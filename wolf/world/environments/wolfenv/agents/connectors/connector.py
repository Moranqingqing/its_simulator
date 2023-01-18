from abc import ABC, abstractmethod



"""
interact with the kernel, read/write.
"""
class Connector(ABC):
    from wolf.world.environments.wolfenv.kernels.wolf_kernel import WolfKernel

    def __init__(self, kernel: WolfKernel):
        self._kernel = kernel

    def reset(self):
        pass

"""
Interface for any connector that needs to know an agent to init or perform calculation.
"""
class AgentListener(ABC):
    from wolf.world.environments.wolfenv.agents.wolf_agent import WolfAgent

    def attach_agent(self, wolf_agent: WolfAgent):
        """
        set the variable agent to agent
        must be override in order to perform extra initialisation using the agent
        see wolf_agent __init__ function where it is called.
        :param wolf_agent:
        :return:
        """
        self.agent = wolf_agent