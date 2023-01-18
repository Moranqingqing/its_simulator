from rltl.distributions.envs_collection import EnvsCollection
from rltl.envs.cartpole import ParametricCartPoleEnv
from rltl.envs.gridworld.envgridworld import EnvGridWorld
from rltl.envs.gridworld.factory import grid0
from rltl.envs.gridworld.geometry import inRectangle
from rltl.envs.mountain_car import ParametricMountainCarEnv
from rltl.envs.lander import ParametricLunarLanderEnv
from rltl.envs.x_env import XEnv


# def slot_filling():
#     d = SlotFillingDistribution()
#     for proba_hangup in [0.0, 0.25, 0.5, 0.75, 1.0]:
#         user_params = {"cerr": -1, "cok": 1, "ser": 0.5, "cstd": 0.2, "proba_hangup": proba_hangup}
#         env_creator = lambda usr_params=user_params: SlotFillingEnv(user_params=usr_params)
#         d.register_env(env_creator, {"user_params": user_params})
#     return d
from wolf.world.environments.wolfenv.gym_wrapper import TrafficEnvSingleAgentGymWrapper


def x(sources=None, targets=None, stochastic=False):
    if sources is None:
        sources = [1, 2, 3]
    if targets is None:
        targets = [1.5, 2.5, 3.5]
    d_sources = EnvsCollection(env_prefix="X_sources")
    d_targets = EnvsCollection(env_prefix="X_targets")
    for x in sources:
        d_sources.register_env(lambda a=x: XEnv(x=a, stochastic=stochastic), {"x": x})
    for x in targets:
        d_targets.register_env(lambda a=x: XEnv(x=a, stochastic=stochastic), {"x": x})
    return d_sources, d_targets


# def x(n=3):
#     d_sources = EnvsCollection(env_prefix="X_{}_sources".format(n))
#     d_targets = EnvsCollection(env_prefix="X_{}_targets".format(n))
#     xs = range(1, n + 1)  # hard to learn the gan
#     for x in xs:
#         d_sources.register_env(lambda a=x: XEnv(x=a), {"x": x})
#         x_target = x + 0.5
#         d_targets.register_env(lambda a=x_target: XEnv(x=a), {"x": x_target})
# >>>>>>> master
#
#     return d_sources, d_targets


# <<<<<<< HEAD
# def x(n=3):
#     d_sources = EnvsCollection(env_prefix="X_{}_sources".format(n))
#     d_targets = EnvsCollection(env_prefix="X_{}_targets".format(n))
#     xs = range(1, n + 1)  # hard to learn the gan
#     for x in xs:
#         d_sources.register_env(lambda a=x: XEnv(x=a), {"x": x})
#         x_target = x + 0.5
#         d_targets.register_env(lambda a=x_target: XEnv(x=a), {"x": x_target})
#
#     return d_sources, d_targets


# =======
# >>>>>>> master
def fast(sources=None, targets=None, horizon=20):
    if sources is None:
        sources = [0.25, 0.50]
    if targets is None:
        targets = [0.33, 0.75]
    d_sources = EnvsCollection(env_prefix="Fast_sources")
    d_targets = EnvsCollection(env_prefix="Fast_targets")
    # for i in [0.01,0.25, 0.50, 1.0, 1.5, 5.]:
    for i in sources:
        length = i * ParametricCartPoleEnv.DEFAULT_LENGTH
        env_creator = lambda l=length: ParametricCartPoleEnv(horizon=horizon, length=l)
        d_sources.register_env(env_creator, {"length": length})
    for i in targets:
        length = i * ParametricCartPoleEnv.DEFAULT_LENGTH
        env_creator = lambda l=length: ParametricCartPoleEnv(horizon=horizon, length=l)
        d_targets.register_env(env_creator, {"length": length})

    return d_sources, d_targets


def gw_wind(sources=None, targets=None, horizon=100, A=None, hole_absorbing=False):
    if sources is None:
        sources = [(0.25, 0.25), (-0.25, -0.25)]
    if targets is None:
        targets = [(0.5, 0.5), (-0.5, -0.5)]
    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    for wind_vector in sources:
        env_creator = lambda w=wind_vector: grid0(hole_absorbing=hole_absorbing, horizon=horizon, A=A,
                                                  wind=lambda x, y: w)()
        d_sources.register_env(env_creator, {"wind_vector": wind_vector})
    for wind_vector in targets:
        env_creator = lambda w=wind_vector: grid0(hole_absorbing=hole_absorbing, horizon=horizon, A=A,
                                                  wind=lambda x, y: w)()
        d_targets.register_env(env_creator, {"wind_vector": wind_vector})

    return d_sources, d_targets


def gw_stochastic2():
    horizon = 10
    lambda_env = lambda std: EnvGridWorld(dim=(2, 2),
                                          std=std, cases=[], horizon=horizon,
                                          penalty_on_move=0.01,
                                          noise_type="uniform",
                                          walls_inside=False,
                                          blocks=[],
                                          init_s=(0.5, 0.5),
                                          actions=[(0.25, 0)],  # , (0, 0.25)],
                                          actions_str=["E"],
                                          wind=None)

    sources = [(0.25, 0.25)]  # , (0, 0.2), (0., 0.3), (0, 0.5)]
    targets = []
    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    for std in sources:
        env_creator = lambda x=std: lambda_env(std=lambda sx, sy: x)
        d_sources.register_env(env_creator, {"std": std})
    for std in targets:
        env_creator = lambda x=std: lambda_env(std=lambda sx, sy: x)
        d_targets.register_env(env_creator, {"std": std})

    return d_sources, d_targets


def gw_wind_distribution(n_sources=10, n_targets=5, horizon=100, A=None, hole_absorbing=False):
    from rltl.distributions.gw_wind_distribution import GridWorldDistribution
    d_sources = GridWorldDistribution(
        env_prefix="GridWorldRandom_sources",
        env_cls_or_factory=lambda wind: grid0(
            wind=lambda x, y: wind,
            horizon=horizon,
            hole_absorbing=hole_absorbing,
            A=A)())
    d_targets = GridWorldDistribution(
        env_prefix="GridWorldRandom_targets",
        env_cls_or_factory=lambda wind: grid0(
            wind=lambda x, y: wind,
            horizon=horizon,
            hole_absorbing=hole_absorbing,
            A=A)())

    for _ in range(n_sources):
        d_sources.sample()

    for _ in range(n_targets):
        d_targets.sample()

    return d_sources, d_targets


def cp_length(sources=None, targets=None, horizon=200):
    if sources is None:
        sources = [0.25, 0.50, 1.0, 1.5, 5.]
    if targets is None:
        targets = [0.33, 0.75, 1.25, 2, 3, 4, 6]
    d_sources = EnvsCollection(env_prefix="Length_sources")
    d_targets = EnvsCollection(env_prefix="Length_targets")
    # for i in [0.01,0.25, 0.50, 1.0, 1.5, 5.]:
    for i in sources:
        length = i * ParametricCartPoleEnv.DEFAULT_LENGTH
        env_creator = lambda l=length: ParametricCartPoleEnv(horizon=horizon, length=l)
        d_sources.register_env(env_creator, {"length": length})
    for i in targets:
        length = i * ParametricCartPoleEnv.DEFAULT_LENGTH
        env_creator = lambda l=length: ParametricCartPoleEnv(horizon=horizon, length=l)
        d_targets.register_env(env_creator, {"length": length})

    return d_sources, d_targets


def cp_gravity(horizon=200):
    d_sources = EnvsCollection(env_prefix="Gravity_sources")
    d_targets = EnvsCollection(env_prefix="Gravity_targets")
    for i in [0.0, 0.25, 0.50, 0.75, 1.0]:
        gravity = i * ParametricCartPoleEnv.DEFAULT_GRAVITY
        env_creator = lambda local_grav=gravity: ParametricCartPoleEnv(horizon=horizon, gravity=local_grav)
        d_sources.register_env(env_creator, {"gravity": gravity})

    for i in [0.125, 0.375, 0.625, 0.875]:
        gravity = i * ParametricCartPoleEnv.DEFAULT_GRAVITY
        env_creator = lambda local_grav=gravity: ParametricCartPoleEnv(horizon=horizon, gravity=local_grav)
        d_targets.register_env(env_creator, {"gravity": gravity})

    return d_sources, d_targets


def ll_density():
    d_sources = EnvsCollection(env_prefix="ll_density_sources")
    d_targets = EnvsCollection(env_prefix="ll_density_targets")
    for density in [0.1, 1.0, 10.0]:
        env_creator = lambda local_density=density: ParametricLunarLanderEnv(density=local_density)
        d_sources.register_env(env_creator, {"density": density})

    for density in [0.05, 0.5, 5.0]:
        env_creator = lambda local_density=density: ParametricLunarLanderEnv(density=local_density)
        d_targets.register_env(env_creator, {"density": density})

    return d_sources, d_targets


def mc_force():
    d_sources = EnvsCollection(env_prefix="mc_force_sources")
    d_targets = EnvsCollection(env_prefix="mc_force_targets")
    # for i in [0.01,0.25, 0.50, 1.0, 1.5, 5.]:
    for i in [0.01, 0.1, 1, 10, 100, 1000]:
        force = i * ParametricMountainCarEnv.DEFAULT_FORCE
        env_creator = lambda l=force: ParametricMountainCarEnv(force=l)
        d_sources.register_env(env_creator, {"force": force})
    for i in [0.05, 0.5, 5, 50, 500]:
        force = i * ParametricMountainCarEnv.DEFAULT_FORCE
        env_creator = lambda l=force: ParametricMountainCarEnv(force=l)
        d_targets.register_env(env_creator, {"force": force})

    return d_sources, d_targets


def mc_max_speed():
    d_sources = EnvsCollection(env_prefix="mc_max_speed_sources")
    d_targets = EnvsCollection(env_prefix="mc_max_speed_targets")
    # for i in [0.01,0.25, 0.50, 1.0, 1.5, 5.]:
    for i in [0.25, 0.5, 0.75, 1, 1.25]:
        max_speed = i * ParametricMountainCarEnv.DEFAULT_MAXSPEED
        env_creator = lambda l=max_speed: ParametricMountainCarEnv(max_speed=l)
        d_sources.register_env(env_creator, {"max_speed": max_speed})
    for i in [0.125, 0.375, 0.675, 0.875, 1.125]:
        max_speed = i * ParametricMountainCarEnv.DEFAULT_MAXSPEED
        env_creator = lambda l=max_speed: ParametricMountainCarEnv(max_speed=l)
        d_targets.register_env(env_creator, {"max_speed": max_speed})

    return d_sources, d_targets


def mc_gravity():
    d_sources = EnvsCollection(env_prefix="mc_gravity_sources")
    d_targets = EnvsCollection(env_prefix="mc_gravity_targets")
    # for i in [0.01,0.25, 0.50, 1.0, 1.5, 5.]:
    for i in [0.001, 0.01, 0.1, 1., 10]:
        gravity = i * ParametricMountainCarEnv.DEFAULT_GRAVITY
        env_creator = lambda l=gravity: ParametricMountainCarEnv(gravity=l)
        d_sources.register_env(env_creator, {"gravity": gravity})
    for i in [0.005, 0.05, 0.5, 5]:
        gravity = i * ParametricMountainCarEnv.DEFAULT_GRAVITY
        env_creator = lambda l=gravity: ParametricMountainCarEnv(gravity=l)
        d_targets.register_env(env_creator, {"gravity": gravity})

    return d_sources, d_targets


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

def to_act(a, step):
    if a == ">":
        return (step, 0)
    if a == "<":
        return (-step, 0)
    if a == "^":
        return (0, -step)
    if a == "v":
        return (0, step)


def gw_generalisation(step=0.25, goal=False, actions=[">"]):
    horizon = 10
    cases = []

    if goal:
        cases.append(((2 - step, 2 - step, 2, 2), 1, 0, True))

    horizon = int((2 / step) * 5)
    lambda_env = lambda dynamics_params: EnvGridWorld(
        dim=(2, 2),
        cases=cases,
        horizon=horizon,
        penalty_on_move=0.01,
        walls_inside=False,
        blocks=[],
        init_s=(step, step),
        actions=[to_act(a, step) for a in actions],
        actions_str=["E"],
        dynamics_params=dynamics_params
    )
    import numpy as np

    angle = np.pi / 4

    sources = [angle * x for x in range(0, 8)]
    targets = [np.pi / 8 + angle * x for x in range(0, 8)]
    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")

    sigma = 0.0

    for rot in sources:
        env_creator = lambda w=rot: lambda_env(
            dynamics_params={
                "type": "rotation",
                "lambda": lambda x, y: (w, sigma)
            })
        d_sources.register_env(env_creator, {"mu_rot": rot})
    for rot in targets:
        env_creator = lambda w=rot: lambda_env(dynamics_params={
            "type": "rotation",
            "lambda": lambda x, y: (w, sigma)
        })
        d_targets.register_env(env_creator, {"mu_rot": rot})

    return d_sources, d_targets


def gw_contextual():
    horizon = 10
    lambda_env = lambda dynamics_params: EnvGridWorld(
        dim=(2, 2),
        cases=[], horizon=horizon,
        penalty_on_move=0.01,
        walls_inside=False,
        blocks=[],
        init_s=(0.5, 0.5),
        actions=[(0.25, 0), (0, 0.25)],
        actions_str=["E"],
        dynamics_params=dynamics_params)

    def create_wind(windy_cases):
        def wind(x, y, cases=windy_cases):
            for case in cases:
                if inRectangle((x, y), case):
                    return (0.125, 0.125)
            return (0, 0)

        return wind

    case0 = (0, 0, 1, 1)
    case1 = (0, 1, 1, 2)
    case2 = (1, 0, 2, 1)
    case3 = (1, 1, 2, 2)

    sources = [[case0], [case1], [case2], [case3]]
    targets = [[case0, case2]]
    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    for windy_cases in sources:
        env_creator = lambda w=windy_cases: lambda_env(dynamics_params={
            "type": "translation",
            "lambda": create_wind(w)
        })
        d_sources.register_env(env_creator, {"cases": windy_cases})
    for windy_cases in targets:
        env_creator = lambda w=windy_cases: lambda_env(dynamics_params={
            "type": "translation",
            "lambda": create_wind(w)
        })
        d_targets.register_env(env_creator, {"cases": windy_cases})

    return d_sources, d_targets


def gw_generalisation_stochastic():
    horizon = 10
    lambda_env = lambda dynamics_params: EnvGridWorld(
        dim=(2, 2),
        cases=[], horizon=horizon,
        penalty_on_move=0.01,

        walls_inside=False,
        blocks=[],
        init_s=(0.5, 0.5),
        actions=[(0.25, 0)],  # , (0, 0.25)],
        actions_str=["E"],
        dynamics_params=dynamics_params)

    # sources = [(0., 0.001), (0, 0.01), (0, 0.1), (0, 1)]  # , (0, 0.2), (0., 0.3), (0, 0.5)]
    sources = [0, 0.05, 0.1, 0.25, 0.5, 1, 10]  # , (0, 0.2), (0., 0.3), (0, 0.5)]
    targets = [0.025, 0.075, 0.175, 0.375, 0.75, 5]
    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    for sigma in sources:
        env_creator = lambda sig=sigma: lambda_env(dynamics_params={
            "type": "rotation",
            "lambda": lambda x, y: (0, sig)
        })
        d_sources.register_env(env_creator, {"sigma_rot": sigma})
    for sigma in targets:
        env_creator = lambda sig=sigma: lambda_env(dynamics_params={
            "type": "rotation",
            "lambda": lambda x, y: (0, sig)
        })
        d_targets.register_env(env_creator, {"sigma_rot": sigma})

    return d_sources, d_targets


def gw_generic_contextual_stochastic(stds, type_dynamics, step=0.25, goal=False, actions=[">"]):
    horizon = 10
    cases = []

    if goal:
        cases.append(((2 - step, 2 - step, 2, 2), 1, 0, True))
    lambda_env = lambda dynamics_params: EnvGridWorld(
        dim=(2, 2),
        cases=cases,
        horizon=horizon,
        penalty_on_move=0.01,
        walls_inside=False,
        blocks=[],
        init_s=(step, step),
        actions=[to_act(a, step) for a in actions],  # , (0, 0.25)],
        actions_str=actions,
        dynamics_params=dynamics_params)
    case0 = (0, 0, 1, 1)
    case1 = (0, 1, 1, 2)
    case2 = (1, 0, 2, 1)
    case3 = (1, 1, 2, 2)

    std0 = stds[0]
    std1 = stds[1]
    std2 = stds[2]
    std3 = stds[3]

    env0 = {
        case0: std0,
        case1: std1,
        case2: std2,
        case3: std3
    }

    env1 = {
        case1: std0,
        case2: std1,
        case3: std2,
        case0: std3
    }

    env2 = {
        case2: std0,
        case3: std1,
        case0: std2,
        case1: std3
    }
    env3 = {
        case3: std0,
        case0: std1,
        case1: std2,
        case2: std3
    }

    env4 = {
        case2: std0,
        case1: std1,
        case0: std2,
        case3: std3
    }

    def create_lambda_dynamics(d):
        def f(sx, sy):
            for case, sigma in d.items():
                if inRectangle((sx, sy), case):
                    return (0, sigma)
            return (0, 0)

        return f

    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    for e in [env0, env1, env2, env3]:
        env_creator = lambda x=e: lambda_env(
            dynamics_params={
                "type": type_dynamics,
                "lambda": create_lambda_dynamics(x)
            })
        d_sources.register_env(env_creator, {"params_std": str(e)})
    for e in [env4]:
        env_creator = lambda x=e: lambda_env(
            dynamics_params={
                "type": type_dynamics,
                "lambda": create_lambda_dynamics(x)
            })
        d_targets.register_env(env_creator, {"params_std": str(e)})

    return d_sources, d_targets


def gw_swap():
    horizon = 10
    lambda_env = lambda dynamics_params: EnvGridWorld(
        dim=(1, 2),
        cases=[], horizon=horizon,
        penalty_on_move=0.01,

        walls_inside=False,
        blocks=[],
        init_s=(0.5, 0.5),
        actions=[(0.25, 0)],  # , (0, 0.25)],
        actions_str=["E"],
        dynamics_params=dynamics_params)
    case0 = (0, 0, 1, 1)
    case1 = (0, 1, 1, 2)
    import numpy as np
    mu0 = 0
    mu1 = np.pi

    env0 = {
        case0: mu0,
        case1: mu1,
    }

    env1 = {
        case0: mu1,
        case1: mu0,
    }

    env2 = {
        case0: mu1,
        case1: mu1,
    }
    sources = [env0, env1]
    targets = [env2]

    sigma = 0.25

    def create_lambda_dynamics(d):
        def f(sx, sy):
            for case, mu in d.items():
                if inRectangle((sx, sy), case):
                    return (mu, sigma)
            return (0, 0)

        return f

    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    for e in sources:
        env_creator = lambda x=e: lambda_env(dynamics_params={
            "type": "rotation",
            "lambda": create_lambda_dynamics(x)
        })
        d_sources.register_env(env_creator, {"e": str(e)})
    for e in targets:
        env_creator = lambda x=e: lambda_env(dynamics_params={
            "type": "rotation",
            "lambda": create_lambda_dynamics(x)
        })
        d_targets.register_env(env_creator, {"e": str(e)})

    return d_sources, d_targets


def gw_contextual_generalisation_stochastic():
    horizon = 10
    goals = [((5, 5, 6, 6), 1, 0, True)]
    lambda_env = lambda std: EnvGridWorld(dim=(6, 6),
                                          std=std, cases=goals, horizon=horizon,
                                          penalty_on_move=0.01,
                                          noise_type="normalized_gaussian",
                                          walls_inside=False,
                                          blocks=[],
                                          init_s=(0.5, 0.5),
                                          actions=[(0.25, 0)],  # , (0, 0.25)],
                                          actions_str=["E"],
                                          wind=None)
    case0 = (0, 0, 1, 1)
    case1 = (0, 1, 1, 2)
    case2 = (1, 0, 2, 1)
    case3 = (1, 1, 2, 2)
    import numpy as np
    std0 = np.array((0.01, 0.01))
    std1 = np.array((0.1, 0.1))
    std2 = np.array((1, 1))
    std3 = np.array((0.0, 0.0))

    env0 = {
        case0: std0,
        case1: std1 * 0.5,
        case2: std2,
        case3: std3 * 0.25
    }

    env1 = {
        case1: std0,
        case2: std1,
        case3: std2 * 0.33,
        case0: std3
    }

    env2 = {
        case2: std0,
        case3: std1 * 0.75,
        case0: std2,
        case1: std3 * 0.33
    }
    env3 = {
        case3: std0,
        case0: std1,
        case1: std2 * 0.66,
        case2: std3
    }

    env4 = {
        case2: std0,
        case1: std1 * 0.18,
        case0: std2 * 0.8,
        case3: std3
    }

    def create_std(d):
        def f(sx, sy):
            for case, std in d.items():
                if inRectangle((sx, sy), case):
                    return std
            return (0, 0)

        return f

    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    for e in [env0, env1, env2, env3]:
        env_creator = lambda x=e: lambda_env(std=create_std(x))
        d_sources.register_env(env_creator, {"params_std": str(e)})
    for e in [env4]:
        env_creator = lambda x=e: lambda_env(std=create_std(x))
        d_targets.register_env(env_creator, {"params_std": str(e)})

    return d_sources, d_targets


def gw_final():
    def lambda_env(dynamics_params, A=None, horizon=None, hole_absorbing=False):
        dim = (9, 9)
        dim_x, dim_y = dim
        if horizon is None:
            horizon = 5 * int(dim_x + dim_y)
            # horizon = 1
        goals = [
            ((6, 8, 9, 9), 1, 0, True),
            ((8, 6, 9, 9), 1, 0, True)
        ]
        # goals = []
        holes = [((2, 2, 3, 3), -1, 0, hole_absorbing),
                 ((3, 2, 4, 3), -1, 0, hole_absorbing),
                 ((1, 7, 4, 8), -1, 0, hole_absorbing),
                 ((7, 1, 9, 2), -1, 0, hole_absorbing),
                 ((5, 6, 6, 7), -1, 0, hole_absorbing)]

        blocks = [
            (3, 3, 4, 4),
            (2, 8, 4, 9),
            (3, 3, 4, 4),
            (6, 2, 7, 4),
            (6, 6, 8, 8)
        ]

        cases = holes + goals
        # start = (1.5, 1.5)
        start = (1, 1)
        if A == None:
            A = [(0., 0.5), (0.5, 0), (-0.5, 0), (0, -0.5)]

        return EnvGridWorld(
            cases=cases,
            dim=dim,
            horizon=horizon,
            penalty_on_move=0,  # 0.01,
            walls_inside=False,
            blocks=blocks,
            init_s=start,
            actions=A,  # , (0, 0.25)],
            dynamics_params=dynamics_params)

    cases = [
        (0, 0, 3, 3), (0, 3, 3, 6), (0, 6, 3, 9),
        (3, 0, 6, 3), (3, 3, 6, 6), (3, 6, 6, 9),
        (6, 0, 9, 3), (6, 3, 9, 6), (6, 6, 9, 9),
    ]

    import numpy as np
    angle = np.pi / 8
    sigmas = [0, 0.05, 0.1, 0.25, 0.5, 1, 10] + [0.025, 0.075, 0.175, 0.375, 0.75, 5]
    mus = [angle * x for x in range(0, 8)] + [np.pi / 8 + angle * x for x in range(0, 8)]

    import numpy as np
    r = np.random.RandomState(0)

    envs = []

    for _ in range(20):
        env = {}
        while len(env) < 9:  # ugly
            key = r.randint(len(cases))
            if key not in env:
                env[cases[key]] = (mus[r.randint(len(mus))], sigmas[r.randint(len(sigmas))])
        envs.append(env)

    def create_lambda_dynamics(d):
        def f(sx, sy):
            for case, (mu, sigma) in d.items():
                if inRectangle((sx, sy), case):
                    return (mu, sigma)
            return (0, 0)

        return f

    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")
    sources = envs[0:int(0.7 * len(envs))]
    targets = envs[int(0.7 * len(envs)):]
    for e in sources:
        env_creator = lambda x=e: lambda_env(dynamics_params={
            "type": "rotation",
            "lambda": create_lambda_dynamics(x)
        })
        d_sources.register_env(env_creator, {"e": str(e)})
    for e in targets:
        env_creator = lambda x=e: lambda_env(dynamics_params={
            "type": "rotation",
            "lambda": create_lambda_dynamics(x)
        })
        d_targets.register_env(env_creator, {"e": str(e)})

    return d_sources, d_targets


#
#
# def gw_final_0():
#     def grid1(std, A=None, horizon=None, hole_absorbing=False):
#         dim = (6, 6)
#         dim_x, dim_y = dim
#         if horizon is None:
#             horizon = 3 * int(dim_x + dim_y)
#         goals = [((5, 5, 6, 6), 1, 0, True)]
#         holes = []
#         # goals = []
#         # holes = [((2, 2, 3, 3), -1, 0, hole_absorbing),
#         #          ((3, 2, 4, 3), -1, 0, hole_absorbing),
#         #          ((1, 7, 4, 8), -1, 0, hole_absorbing),
#         #          ((7, 1, 9, 2), -1, 0, hole_absorbing),
#         #          ((6, 7, 5, 6), -1, 0, hole_absorbing)]
#         #
#         # blocks = [
#         #     (3, 3, 4, 4),
#         #     (2, 8, 4, 9),
#         #     (3, 3, 4, 4),
#         #     (7, 2, 6, 4),
#         #     (6, 6, 8, 8)
#         # ]
#
#         cases = holes + goals
#         # start = (1.5, 1.5)
#         start = (1.5, 1.5)
#         if A == None:
#             A = [(0., 0.5), (0.5, 0)]  # , (-0.5, 0), (0, -0.5)]
#
#         return EnvGridWorld(dim=dim, cases=cases, horizon=horizon,
#                             noise_type="normalized_gaussian",
#                             std=std,
#                             blocks=[],  # blocks,
#                             penalty_on_move=0.01,
#                             walls_inside=False,
#                             init_s=start, actions=A,
#                             wind=None)
#
#     cases = [
#         (0, 0, 2, 2), (0, 2, 2, 4), (0, 4, 4, 6),
#         (2, 0, 4, 2), (2, 2, 4, 4), (2, 4, 4, 6),
#         (4, 0, 6, 2), (4, 2, 6, 4), (4, 4, 6, 6),
#     ]
#
#     stds = [(x * y, x * y) for x in [0.25, 0.5, 1] for y in [0.01, 0.1, 1]]
#
#     import numpy as np
#     r = np.random.RandomState(0)
#
#     envs = []
#
#     for _ in range(20):
#         env = {}
#         while len(env) < 9:  # ugly
#             key = r.randint(len(cases))
#             if key not in env:
#                 env[cases[key]] = stds[r.randint(len(stds))]
#         envs.append(env)
#
#     def create_std(d):
#         def f(sx, sy):
#             for case, std in d.items():
#                 if inRectangle((sx, sy), case):
#                     return std
#             return (0, 0)
#
#         return f
#
#     d_sources = EnvsCollection(env_prefix="Gridworld_sources")
#     d_targets = EnvsCollection(env_prefix="Gridworld_targets")
#     sources = envs[0:int(0.7 * len(envs))]
#     targets = envs[int(0.7 * len(envs)):]
#     for e in sources:
#         env_creator = lambda w=e: grid1(std=create_std(w))
#         d_sources.register_env(env_creator, {"std": str(e)})
#     for e in targets:
#         env_creator = lambda w=e: grid1(std=create_std(w))
#         d_targets.register_env(env_creator, {"std": str(e)})
#
#     return d_sources, d_targets


def gw_unit_test_0():
    horizon = 10
    lambda_env = lambda dynamics_params: EnvGridWorld(
        dim=(1, 1),
        cases=[], horizon=horizon,
        penalty_on_move=0.01,

        walls_inside=False,
        blocks=[],
        init_s=(0.5, 0.5),
        actions=[(0.25, 0)],  # , (0, 0.25)],
        actions_str=["E"],
        dynamics_params=dynamics_params)

    d_sources = EnvsCollection(env_prefix="Gridworld_sources")
    d_targets = EnvsCollection(env_prefix="Gridworld_targets")

    env_creator = lambda: lambda_env(dynamics_params={
        "type": "rotation",
        "lambda": lambda x, y: (0, 1.25)
    })
    d_sources.register_env(env_creator, "gw_unit_test_0")

    return d_sources, d_targets


def traffic():
    simulator = "traci"
    sim_params = {
        "restart_instance": True,
        "sim_step": 1,
        "print_warnings": False,
        "render": False,
    }
    env_state_params = None
    group_agents_params = None
    multi_agent_config_params = {
        "name": "shared_policy",
        "params": {}
    }
    agents_params = {
        "name": "all_the_same",
        "params": {
            "global_reward": False,
            "default_policy": None,

            "action_params": {
                "name": "ExtendChangePhaseConnector",
                "params": {},
            },
            # "obs_params": {
            #     "name": "TDTSEConnector",
            #     "params": {
            #         "obs_params": {
            #             "num_history": 1,
            #             "detector_position": [5, 100],
            #         },
            #         "raw_obs": True,
            #         "phase_channel": True
            #     }
            # },
            "obs_params": {
                "name": "QueueObservationConnector",
                "params": {
                    "obs_params": {
                        "num_history": 60,
                        "detector_position": [5, 100],
                    },
                    "raw_obs": True,
                    "phase_channel": True
                }
            },
            "reward_params": {
                "name": "QueueRewardConnector",
                "params": {
                    "stop_speed": 2
                }
            }
        }
    }
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    def lambda_env(mu, sigma):
        env = WolfEnv.create_env(
            cls=SimpleGridEnv,

            agents_params=agents_params,
            env_state_params=env_state_params,
            group_agents_params=group_agents_params,
            multi_agent_config_params=multi_agent_config_params,
            n=1,
            m=1,
            inflow_params={"WE": (mu, sigma),  # (0.2, 1.),
                           "EW": (0., 0.),
                           "NS": (0., 0.),
                           "SN": (0., 0.)},
            horizon=500,
            detector_params={'positions': [-5, -100], 'frequency': 100},
            simulator=simulator,
            sim_params=sim_params)
        return TrafficEnvSingleAgentGymWrapper(env)

    d_sources = EnvsCollection(env_prefix="Traffic_sources")
    d_targets = EnvsCollection(env_prefix="Traffic_targets")

    for mu, sigma in [(0.05, 0.),(0.1, 0.),(0.2, 0.), (0.5, 0.), (1., 0)]:
        env_creator = lambda mu_=mu, sigma_=sigma: lambda_env(mu=mu_, sigma=sigma_)
        d_sources.register_env(env_creator, "traffic_{}_{}".format(mu, sigma))

    # for mu, sigma in [(0.7, 1.)]:
    #     env_creator = lambda mu_=mu, sigma_=sigma: lambda_env(mu=mu_, sigma=sigma_)
    #     d_targets.register_env(env_creator, "traffic_{}_{}".format(mu, sigma))

    return d_sources, d_targets
