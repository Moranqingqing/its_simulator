from collections import defaultdict

from rltl.envs.gridworld.factory import grid1
from rltl.main.dqn_agent import FeedForwardQModel
from rltl.main.policy import StaticMaxPolicy
from rltl.utils.envs_collection_factory import *


class Registry:

    def __init__(self):
        self.d = defaultdict(dict)

    def get_env(self, name):
        return self.d["envs"][name]

    def get_envs_collection(self, name):
        return self.d["envs_collection"][name]

    def get_q_model_constructor(self, name):
        return self.d["q_model_constructor"][name]


R = Registry()

R.d["policy"]["StaticMaxPolicy"] = StaticMaxPolicy(action_space=None)

R.d["envs_collection"]["debug_high_length"] = lambda: cp_length(sources=[5], targets=[])
R.d["envs_collection"]["cp_gravity"] = lambda: cp_gravity()
R.d["envs_collection"]["mc_force_collection"] = lambda: mc_force()
R.d["envs_collection"]["fast"] = lambda: fast()
R.d["envs_collection"]["mc_max_speed_collection"] = lambda: mc_max_speed()
R.d["envs_collection"]["mc_gravity_collection"] = lambda: mc_gravity()
# R.d["envs_collection"]["x"] = lambda: x(n=3)
# R.d["envs_collection"]["x_5"] = lambda: x(n=5)
R.d["q_model_constructor"]["FeedForwardQModel"] = FeedForwardQModel

R.d["envs_collection"]["cp_length"] = lambda: cp_length()

R.d["envs_collection"]["cp_length_easy"] = lambda: cp_length(
    sources=[0.25, 0.50, 1.0, 1.5, 2., 2.5, 3.],
    targets=[0.33, 0.75, 1.25, 1.75, 2.25, 2.75])

R.d["envs_collection"]["cp_generalisation"] = lambda: cp_length(
    sources=[0.25, 3.],
    targets=[1.25, 1.75])

R.d["envs_collection"]["x_generalisation"] = lambda: x(
    sources=[0, 1],
    targets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

R.d["envs_collection"]["x_stochastic"] = lambda: x(
    sources=[5],
    targets=[5],
    stochastic=True)

R.d["envs_collection"]["x_not_stochastic"] = lambda: x(
    sources=[5],
    targets=[5],
    stochastic=False)

#########################################
### gridworld collections
#########################################

# R.d["envs_collection"]["gw_wind_generalisation"] = lambda: gw_wind(
#     sources=[(-1 + 0.25 * x, -1 + 0.25 * x) for x in range(9)],
#     targets=[(-1.125 + 0.25 * x, -1.125 + 0.25 * x) for x in range(9)],
#     # targets=[(-0.125, 0), (0.125, 0), (0, 0.125), (0, -0.125), (-0.5, 0), (0.5, 0), (0, 0.5), (0, -0.5)],
#     # A=[(0, 0.5)], #, (0.5, 0)]
#     A=[(0.5, 0), (0, 0.5), (-0.5, 0), (0, -0.5)]
# )

# R.d["envs_collection"]["gw_random"] = lambda: gw_wind_distribution(100, 25, A=[(0.5, 0), (0, 0.5)])
#
# R.d["envs_collection"]["gw_wind_final"] = lambda: gw_wind(
#     sources=[(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)],
#     targets=[(0.5, 0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5)],
#     A=[(1, 0), (0, 1), (-1, 0), (0, -1)]
# )

envs_collection = R.d["envs_collection"]

envs_collection["gw_contextual"] = lambda: gw_contextual()
envs_collection["gw_generalisation_stochastic"] = lambda: gw_generalisation_stochastic()
envs_collection["gw_generalisation"] = lambda: gw_generalisation()
envs_collection["gw_contextual_stochastic"] = lambda: \
    gw_generic_contextual_stochastic([0.15, 0.5, 1.25, 0.05], "rotation")
envs_collection["gw_discrete_contextual_stochastic"] = lambda: \
    gw_generic_contextual_stochastic([0.15, 0.5, 1.25, 0.05], "discrete_rotation")
envs_collection["gw_easy_contextual_stochastic"] = lambda: \
    gw_generic_contextual_stochastic([0, 0, 1, 0], "rotation")
envs_collection["gw_easy_discrete_contextual_stochastic"] = lambda: \
    gw_generic_contextual_stochastic([0, 0, 1.5, 0], "discrete_rotation")
envs_collection["gw_contextual_generalisation_stochastic"] = lambda: gw_contextual_generalisation_stochastic()
envs_collection["gw_swap"] = lambda: gw_swap()
envs_collection["gw_final"] = lambda: gw_final()
envs_collection["gw_unit_test_0"] = lambda: gw_unit_test_0()
envs_collection["wolfenv"] = lambda: traffic()

envs_collection["4_rooms_stochastic"] = lambda: gw_generic_contextual_stochastic(
    stds=[0.15, 0.5, 1.25, 0.05],
    type_dynamics="rotation",
    step=0.2,
    goal=True,
    actions=[">", "v", "<", "^"]
)

envs_collection["rl_rotating_room"] = lambda: gw_generalisation(
    step=0.2,
    goal=True,
    actions=[">", "v", "<", "^"]
)
