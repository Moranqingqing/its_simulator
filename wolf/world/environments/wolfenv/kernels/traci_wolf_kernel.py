from wolf.world.environments.wolfenv.kernels.wolf_kernel import WolfKernel


class TraciWolfKernel(WolfKernel):
    """"""

    def __init__(self, flow_kernel, flow_network, sim_params, tl_params, controlled_nodes):
        super().__init__(flow_kernel, flow_network, sim_params, tl_params, controlled_nodes)
        pass

    # implement methods that are not implemented in TrafficNetworkData
