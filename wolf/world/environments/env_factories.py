from wolf.world.environments.ctm.ctm_env import CtmEnv, EAST, NORTH, SOUTH
from wolf.utils.math import recursive_dict_update

import numpy as np


def simple_grid(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        n=env_config["n"],
        m=env_config["m"],
        row_proba=env_config["row_proba"],
        col_proba=env_config["col_proba"],
        inflow_type=env_config["inflow_type"],
        inflow_params=env_config["inflow_params"],
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def generic_grid(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import GenericGridEnv

    env = WolfEnv.create_env(
        cls=GenericGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        inflow_type=env_config["inflow_type"],
        inflow_params=env_config["inflow_params"],
        simulator=env_config["simulator"],
        vehicles_params=env_config["vehicles_params"],
        env_params=env_config["env_params"],
        sim_params=env_config["sim_params"],
        net_params=env_config["net_params"],
        initial_config_params=env_config["initial_config_params"],
        detector_params=env_config["detector_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def test0(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        n=1,
        m=2,
        inflow_params={'WE': (1, 0), 'EW': (0, 0), 'NS': (0, 0), 'SN': (0, 0)},
        horizon=200,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        tl_params=env_config.get("tl_params", {}),
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def test0_1(env_config):
    """
    Adapted from test0 config.
    Moves upstream loop detectors backwards, right upto the previous intersection.
    Uses platoon inflows.

    Args:
        env_config (dict): Environment config.

    Returns:
        wolf.world.environments.wolfenv.grid_env.SimpleGridEnv: SimpleGridEnv object.
    """
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    tl_params = {
            "ALL": {
                "params": {
                    "initialization": "fixed",
                },
            }
        }
    tl_params = recursive_dict_update(tl_params, env_config.get("tl_params", {}))

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        n=1,
        m=2,
        inflow_type='platoon',
        inflow_params={'WE': (10, 20), 'EW': (0, 30), 'NS': (0, 30), 'SN': (0, 30)},
        horizon=200,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        detector_params={"positions": [-5, 5], "frequency": 100},
        tl_params=tl_params,
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def test0_2(env_config):
    """
    Single intersection version of test0_1.

    Args:
        env_config (dict): Environment config.

    Returns:
        wolf.world.environments.wolfenv.grid_env.SimpleGridEnv: SimpleGridEnv object.
    """
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    tl_params = {
            "ALL": {
                "params": {
                    "initialization": "fixed",
                },
            }
        }
    tl_params = recursive_dict_update(tl_params, env_config.get("tl_params", {}))

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        n=1,
        m=1,
        inflow_type='platoon',
        inflow_params={'WE': (10, 20), 'EW': (0, 30), 'NS': (0, 30), 'SN': (0, 30)},
        horizon=200,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        detector_params={"positions": [-5, 5], "frequency": 100},
        tl_params=tl_params,
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def test0_3(env_config):
    """
    Adapted from test0_1 config.
    Multi-directional wolfenv.

    Args:
        env_config (dict): Environment config.

    Returns:
        wolf.world.environments.wolfenv.grid_env.SimpleGridEnv: SimpleGridEnv object.
    """

    # inflow_params of {NS: (30, 0)} is applied to both south-bound lanes, which was not the intention.
    # TODO (parth): change inflow_params such that inflows can be applied to each lane individually.

    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    tl_params = {
            "ALL": {
                "params": {
                    "initialization": "fixed",
                },
            }
        }
    tl_params = recursive_dict_update(tl_params, env_config.get("tl_params", {}))

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        n=1,
        m=2,
        inflow_type='platoon',
        inflow_params={'WE': (30, 0), 'EW': (0, 30), 'NS': (30, 0), 'SN': (0, 30)},
        horizon=500,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        detector_params={"positions": [-5, 5], "frequency": 100},
        tl_params=tl_params,
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def test1(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        group_agents_params=env_config["group_agents_params"],
        n=1,
        m=2,
        inflow_params={'WE': (0.7, 0), 'EW': (0.05, 0), 'NS': (0.05, 0), 'SN': (0.05, 0)},
        horizon=500,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        tl_params=env_config["tl_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def grid_master_slaves(env_config, n, base_len):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv
    col_inner_lenghts = [(i + 1) * base_len for i in range(n - 1)]

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        col_inner_lengths=col_inner_lenghts,
        row_inner_lengths=[],
        inflow_params={'WE': (0.5, 0), 'EW': (0.025, 0), 'NS': (0.025, 0), 'SN': (0.025, 0)},
        horizon=500,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def test3_1(env_config):
    """
    Adapted from test0 config, but with single intersection.
    Moves upstream loop detectors backwards, right upto the previous intersection.
    Uses platoon inflows.

    Args:
        env_config (dict): Environment config.

    Returns:
        wolf.world.environments.wolfenv.grid_env.SimpleGridEnv: SimpleGridEnv object.
    """
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    tl_params = {
            "ALL": {
                "params": {
                    "initialization": "fixed",
                },
            }
        }
    tl_params = recursive_dict_update(tl_params, env_config.get("tl_params", {}))

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        n=1,
        m=1,
        inflow_type='platoon',
        inflow_params={'WE': (20, 0), 'EW': (0, 30), 'NS': (0, 30), 'SN': (0, 30)},
        horizon=500,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        detector_params={"positions": [-5, 5], "frequency": 100},
        tl_params=tl_params,
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def test3_2(env_config):
    """
    Multi-directional wolfenv version of test3_1.

    Args:
        env_config (dict): Environment config.

    Returns:
        wolf.world.environments.wolfenv.grid_env.SimpleGridEnv: SimpleGridEnv object.
    """
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

    tl_params = {
            "ALL": {
                "params": {
                    "initialization": "fixed",
                },
            }
        }
    tl_params = recursive_dict_update(tl_params, env_config.get("tl_params", {}))

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        n=1,
        m=1,
        inflow_type='platoon',
        inflow_params={'WE': (30, 0), 'EW': (0, 30), 'NS': (20, 10), 'SN': (0, 30)},
        horizon=500,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        detector_params={"positions": [-5, 5], "frequency": 100},
        tl_params=tl_params,
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def grid_gaussian_master_slaves(env_config, n, base_len):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv
    col_inner_lenghts = [(i + 1) * base_len for i in range(n - 1)]

    env = WolfEnv.create_env(
        cls=SimpleGridEnv,
        agents_params=env_config["agents_params"],
        env_state_params=env_config["env_state_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        col_inner_lengths=col_inner_lenghts,
        row_inner_lengths=[],
        inflow_params={'WE': (0.5, 0.7), 'EW': (0.025, 0.1), 'NS': (0.025, 0.1), 'SN': (0.025, 0.1)},
        horizon=500,
        simulator=env_config["simulator"],
        sim_params=env_config["sim_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
    )
    return env


def real_world_network(env_config, horizon=6300):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.real_net_env import RealWorldNetworkEnv

    tl_params = {
        "ALL": {
            "name": "second_based",
            "params": {
                "default_constraints": {"min_time": 10, "max_time": 60},
                "initialization": "random",
            }
        }
    }
    if "tl_params" in env_config:
        tl_params = recursive_dict_update(tl_params, env_config["tl_params"])

    vehicles_params = {
        "min_gap": 2.5,
        "max_speed": 30,
        "decel": 7.5,
        "speed_mode": "right_of_way",
    }

    # horizon = env_config.get('horizon')
    # horizon is required for real_world_network. 
    # Previously, if horizon was not specified in the config, it was assumed to be 6300 in this method.
    # Temporarily uncomment below line, if using rollout.py to evaluate a model that was trained on an older config in which horizon was not specified.
    # horizon = 6300
    # assert horizon, 'horizon is required for real_world_network.'

    env_params = {
        "horizon": horizon,
        "additional_params": {
            "target_velocity": 50,
            "switch_time": 3,
            "num_observed": 2,
            "discrete": False,
            "tl_type": "actuated",
            "num_local_edges": 4,
            "num_local_lights": 4,
        },
    }

    # custom initialization is not implemented yet
    # initial_config_params = {"spacing": 'custom', "shuffle": True}
    initial_config_params = {}

    if "net_params" not in env_config:
        net_params = {
            # "template":
            #   "net": "wolf/sumo_net/shenzhen_3x3/shenzhen_3by3_loops_modified_12int_center.net.xml"
            #   "rou": "wolf/sumo_net/shenzhen_3x3/flow_turns_rou_0506.rou.xml"
            #   "vtype": "wolf/sumo_net/shenzhen_3x3/flow_turns_rou_0506.rou.xml"
            #   "add": "wolf/sumo_net/shenzhen_3x3/shenzhen_3by3_loopfile_center.xml"
            "template": {
                "net": "wolf/sumo_net/wujiang/china_net_5p_ups_LD_noUturn_police.net.xml",
                "rou": "wolf/sumo_net/wujiang/china_flows_1hr45min_noUturn_ups.rou.xml",
                "vtype": "wolf/sumo_net/wujiang/china_flows_1hr45min_noUturn_ups.rou.xml",
                "add": "wolf/sumo_net/wujiang/china_net_5p_ups_loop_detectors.xml",
            },
            # "controlled_tls" : ['center1', 'center2', 'center3', 'center4', 'center5', 'center6', 'center7', 'center8', 'center9']
            # "controlled_tls" : ['center1', 'center2', 'center3']
            "controlled_tls": ['main_center']
        }
    else:
        net_params = env_config["net_params"]

    env = WolfEnv.create_env(
        cls=RealWorldNetworkEnv,
        tl_params=tl_params,
        agents_params=env_config["agents_params"],
        group_agents_params=env_config["group_agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        vehicles_params=vehicles_params,
        env_params=env_params,
        sim_params=env_config["sim_params"],
        detector_params=None,
        net_params=net_params,
        initial_config_params=initial_config_params,
        env_state_params=env_config["env_state_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
    )

    return env


real_or_fake_keys = ["real", "fake"]
stationary_keys = ["stationary", "rushes", "random_rushes"]
demand_symmetry_keys = ["unique", "master_slaves", "full", "mixed"]
demand_distribution_keys = ["platoon", "bernoulli", "gaussian", "poisson"]
layout_symmetry_keys = [True, False]


def benchmark_env(env_config,
                  real_or_fake="real",
                  demand_distribution=None,
                  stationary=None,
                  layout_symmetry=None,
                  demand_symmetry=None,
                  n=None,
                  m=None):
    """
    :param real_or_fake:  real of fake
    :param demand_distribution: platton, gaussian, bernouilli
    :param stationary: stationary, rushes, random_rushes
    :param layout_symmetry: true/false
    :param demand_symmetry: unique, master_slaves, full, mixed
    :param n: >0
    :param m: >0, none if real network
    :param env_config: dictionary
    :return: an instance of an environment
    """

    check("real_or_fake", real_or_fake, real_or_fake_keys)
    if real_or_fake == "fake":
        check("stationary", stationary, stationary_keys)
        check("layout_symmetry", layout_symmetry, layout_symmetry_keys)
        check("demand_symmetry", demand_symmetry, demand_symmetry_keys)
        check("demand_distribution", demand_distribution, demand_distribution_keys)

    if real_or_fake == "real":
        if n is not None:
            raise NotImplemented("Size of real network is fixed for now, cf trello ticket")
        else:
            return real_world_network(env_config, horizon=6300)
    elif real_or_fake == "fake":
        from wolf.world.environments.wolfenv.wolf_env import WolfEnv
        from wolf.world.environments.wolfenv.grid_env import SimpleGridEnv

        if stationary == "stationary":
            pass
        else:
            raise NotImplemented()

        if demand_distribution == "platoon":
            inflow_type = 'platoon'
            inflow_params = {'WE': (12, 20), 'EW': (3, 30), 'NS': (3, 30), 'SN': (3, 30)}
        elif demand_distribution == "gaussian":
            inflow_type = "gaussian"
            inflow_params = {'WE': (0.5, 0.7), 'EW': (0.025, 0.1), 'NS': (0.025, 0.1), 'SN': (0.025, 0.1)}
        elif demand_distribution == "bernoulli":
            inflow_type = "gaussian"
            inflow_params = {'WE': (0.5, 0), 'EW': (0.1,0), 'NS': (0.1,0), 'SN': (0.1,0)}
        elif demand_distribution == "poisson":
            inflow_type = "poisson"
            inflow_params = {'WE': (0.2, 0), 'EW': (0.3, 0),'NS': (0.2, 0), 'SN': (0.1, 0)}
            # Second parameter is not used, included for consistency with other cases
        else:
            raise ValueError()

        if demand_symmetry == "unique":
            inflow_params["EW"] = None
            inflow_params["NS"] = None
            inflow_params["SN"] = None
        elif demand_symmetry == "master_slaves":
            pass
        elif demand_symmetry == "full":
            inflow_params["EW"] = inflow_params["WE"]
            inflow_params["NS"] = inflow_params["WE"]
            inflow_params["SN"] = inflow_params["WE"]
        elif demand_symmetry == "mixed":
            # keep the parameters as they are
            pass
        else:
            raise ValueError()

        base_len = 300
        if layout_symmetry is None:
            raise ValueError()
        elif layout_symmetry:
            col_inner_lengths = [(i + 1) * base_len for i in range(m - 1)]
            row_inner_lengths = [(i + 1) * base_len for i in range(n - 1)]
        else:
            col_inner_lengths = [] if m == 1 else (m - 1) * [base_len]
            row_inner_lengths = [] if n == 1 else (n - 1) * [base_len]

        env = WolfEnv.create_env(
            cls=SimpleGridEnv,
            agents_params=env_config["agents_params"],
            env_state_params=env_config["env_state_params"],
            group_agents_params=env_config["group_agents_params"],
            multi_agent_config_params=env_config["multi_agent_config_params"],
            col_inner_lengths=col_inner_lengths,
            row_inner_lengths=row_inner_lengths,
            short_length=1000,
            long_length=1000,
            inflow_type=inflow_type,
            detector_params={'positions': [-5, -100], 'frequency': 100},
            n=n,
            m=m,
            inflow_params=inflow_params,
            horizon=1000,
            simulator=env_config["simulator"],
            sim_params=env_config["sim_params"],
            tl_params=env_config.get("tl_params", {}),
            action_repeat_params=env_config.get("action_repeat_params", None),
        )
        return env
    else:
        raise ValueError()


def check(str_key, key, keys):
    if key not in keys:
        raise Exception("{} \"{}\" should be in  {}".format(str_key, key, keys))


def parse_benchmark_params(name):
    stationary_keys = ["S", "Ru", "RuR"]
    layout_symmetry_keys = ["Sy", "As"]
    demand_symmetry_keys = ["U", "MS", "Fu"]
    demand_distribution_keys = ["P", "B", "G", "S"]

    d = {
        "R": "real",
        "F": "fake",
        "S": "stationary",
        "Ru": "rushes",
        "RuR": "random_rushes",
        "Sy": True,
        "As": False,
        "U": "unique",
        "MS": "master_slaves",
        "Fu": "full",
        "M": "mixed",
        "P": "platoon",
        "B": "bernoulli",
        "G": "gaussian",
        "S": "poisson"
    }

    name_split = name.split("_")
    real_of_fake = name_split[0]
    if real_of_fake == "F":
        if len(name_split) < 6:
            raise Exception("Fake network should have 6 elements")

        layout_symmetry = name_split[1]  # SY or ASY
        nxm = name_split[2]
        stationary = name_split[3]  # S NS RNS
        demand_symmetry = name_split[4]  #
        demand_distribution = name_split[5]

        check("Layout symmetry key", layout_symmetry, layout_symmetry_keys)
        check("Stationary key", stationary, stationary_keys)
        check("Demand symmetry key", demand_symmetry, demand_symmetry_keys)
        check("Demand distribution key", demand_distribution, demand_distribution_keys)

        try:
            n, m = nxm.split("x")
            n = int(n)
            m = int(m)
        except:
            raise Exception("n and m are wrongly formatted, {} should be nxm".format(name_split[2]))

        return {
            "real_or_fake": d[real_of_fake],
            "demand_distribution": d[demand_distribution],
            "stationary": d[stationary],
            "layout_symmetry": d[layout_symmetry],
            "demand_symmetry": d[demand_symmetry],
            "n": n,
            "m": m
        }
    elif real_of_fake == "R":
        if len(name_split) == 1:
            return {"real_or_fake": "real"}
        else:
            n = name_split[1]
            return {"real_or_fake": d[real_of_fake], "n": n}
    else:
        raise Exception("First letter must be R (for real env) or F (for fake env), instead name is {}".format(name))


########################################################################################################################
########################################################################################################################
#######################################           CTM ENV         ######################################################
########################################################################################################################
########################################################################################################################


def ctm_test0(env_config):
    """
    With occupation reward, the agent should optimally reach -8.57 if the light are already green on main arterial.
    Agent reach almost-optimality after 100k step
    :param env_config:
    :return:
    """

    def sample_cars(x, cardinal, k):
        if cardinal == EAST:
            return 5
        else:
            return 0

    env = CtmEnv(sample_cars=sample_cars,
                 r_roads=1, c_roads=1,
                 h_offset=3, w_offset=3,
                 horizon=200, local_obs_radius=2,
                 **env_config)

    print("optimal value=", -env.horizon * 5 * (env.w_offset * (env.c_roads + 1)) / (
            env.road_cell_number * env.cell_max_capacity))  # -8.57
    return env


def ctm_test0_1(env_config):
    """
    With occupation reward, the agent should optimally reach -8.57 if the light are already green on main arterial.
    Agent reach almost-optimality after 100k step
    :param env_config:
    :return:
    """

    def sample_cars(x, cardinal, k):
        if cardinal == EAST:
            return 2
        else:
            return 1

    env = CtmEnv(sample_cars=sample_cars,
                 r_roads=1, c_roads=1,
                 h_offset=3, w_offset=3,
                 max_green_time=50,
                 min_green_time=5,
                 horizon=500, local_obs_radius=2,
                 **env_config)

    return env


def ctm_test1(env_config):
    def sample_cars(x, cardinal, k):
        if cardinal == EAST:
            return 5
        else:
            return 0

    env = CtmEnv(sample_cars=sample_cars,
                 r_roads=1, c_roads=1,
                 h_offset=3, w_offset=3,
                 horizon=200, local_obs_radius=2,
                 **env_config)

    print("optimal value=", -env.horizon * 5 * (env.w_offset * (env.c_roads + 1)) / (
            env.road_cell_number * env.cell_max_capacity))
    return env


def ctm_test2(env_config):
    def sample_cars(x, cardinal, k):
        if cardinal == EAST:
            return 5  # np.random.randint(0, 10)
        elif cardinal == NORTH:
            return 0  # np.random.randint(0, 1)
        elif cardinal == SOUTH:
            return 0  # np.random.randint(0, 1)
        else:
            return 0

    env = CtmEnv(sample_cars=sample_cars,
                 horizon=200,
                 r_roads=1, c_roads=2,
                 h_offset=3, w_offset=3,
                 local_obs_radius=4,
                 **env_config)

    print("optimal value=",
          -env.horizon * 5 * (env.w_offset * (env.c_roads + 1)) / (env.road_cell_number * env.cell_max_capacity))

    return env


def ctm_test3(env_config):
    def sample_cars(x, cardinal, k):
        if cardinal == EAST:
            return 5
        else:
            return 0

    env = CtmEnv(horizon=100, sample_cars=sample_cars, local_obs_radius=3, r_roads=1, c_roads=1, h_offset=3, w_offset=3,
                 **env_config)
    return env


def ctm_test4(env_config):
    """
    APEX_DQN does -44 in 200k steps for gamma=0.99 and target_network_size=1000, does not change after (tested until 2m steps)
    Does way worst for gama=0.9, does a bit worst for gamma=0.999
    :param env_config:
    :return:
    """

    def sample_cars(x, cardinal, k):
        if cardinal == EAST:
            return np.random.randint(0, 10)
        elif cardinal == NORTH:
            return 1 if np.random.random() < 0.1 else 0
        elif cardinal == SOUTH:
            return 1 if np.random.random() < 0.1 else 0
        else:
            return 0

    env = CtmEnv(sample_cars=sample_cars,
                 horizon=500,
                 r_roads=1, c_roads=2,
                 h_offset=5, w_offset=5,
                 local_obs_radius=4,
                 **env_config)

    return env


def ctm_test5(env_config):
    def sample_cars(x, cardinal, k):
        if cardinal == EAST:
            return np.random.randint(0, 12)
        elif cardinal == NORTH:
            return 1 if np.random.random() < 0.05 else 0
        elif cardinal == SOUTH:
            return 1 if np.random.random() < 0.05 else 0
        else:
            return 0

    env = CtmEnv(sample_cars=sample_cars,
                 horizon=500,
                 max_green_time=50,
                 min_green_time=4,
                 r_roads=1, c_roads=3,
                 h_offset=7, w_offset=7,
                 local_obs_radius=4,
                 **env_config)

    return env


def car_following_test(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.car_following_env import ClosedRoadNetCarFollowing

    # TODO: Remove this once Aimsun and Wolf versions of Flow are compatible
    #from wolf.world.environments.wolfenv.car_following_env_aimsun import ClosedRoadNetCarFollowing as ClosedRoadAimsun

    cls = ClosedRoadNetCarFollowing if env_config['simulator'] == 'traci' else \
          ClosedRoadAimsun

    env = WolfEnv.create_env(
        cls=cls,
        agents_params=env_config["agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        group_agents_params=env_config["group_agents_params"],
        sim_params=env_config['sim_params'],
        env_state_params=env_config["env_state_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
        record_flag=env_config['record_flag'], # FIXME: Temp soulutions, metrics record should connect to ray
        reward_folder=env_config['reward_folder'], # FIXME: Temp soulutions, metrics record should connect to ray 
        simulator=env_config["simulator"]
    )

    return env

def car_following_eval(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.car_following_env import ClosedRoadNetCarFollowingEval

    env = WolfEnv.create_env(
        cls=ClosedRoadNetCarFollowingEval,
        agents_params=env_config["agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        group_agents_params=env_config["group_agents_params"],
        sim_params=env_config['sim_params'],
        env_state_params=env_config["env_state_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
        simulator=env_config["simulator"],
        trial_id=env_config.get("trial_id", None)
    )

    return env

def car_following_eval1(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.car_following_env import ClosedRoadNetCarFollowingEval1

    env = WolfEnv.create_env(
        cls=ClosedRoadNetCarFollowingEval1,
        agents_params=env_config["agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        group_agents_params=env_config["group_agents_params"],
        sim_params=env_config['sim_params'],
        env_state_params=env_config["env_state_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
        simulator=env_config["simulator"],
        trial_id=env_config.get("trial_id", None)
    )

    return env

def car_following_qew(env_config):
    from wolf.world.environments.wolfenv.wolf_env import WolfEnv
    from wolf.world.environments.wolfenv.car_following_env_aimsun import QEWCarFollowing

    return WolfEnv.create_env(
        cls=QEWCarFollowing,
        agents_params=env_config["agents_params"],
        multi_agent_config_params=env_config["multi_agent_config_params"],
        group_agents_params=env_config["group_agents_params"],
        sim_params=env_config['sim_params'],
        env_state_params=env_config["env_state_params"],
        action_repeat_params=env_config.get("action_repeat_params", None),
        record_flag=env_config['record_flag'], # FIXME: Temp soulutions, metrics record should connect to ray
        reward_folder=env_config['reward_folder'], # FIXME: Temp soulutions, metrics record should connect to ray 
        simulator=env_config["simulator"]
    )
