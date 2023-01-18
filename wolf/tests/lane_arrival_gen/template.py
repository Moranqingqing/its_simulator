"""
Template script for testing classes derived from LaneArrivals
(The classes being tested are defined in the lane_arrival_gen module.)
The environment is a grid Traffic Light Control environment based on
GenericGridEnv, but simplified slightly.
"""

from wolf.world.environments.wolfenv.wolf_env import WolfEnv

# Traffic light actions, and utils
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND, CHANGE
from wolf.utils.math import *

# The module being tested
import wolf.world.environments.wolfenv.lane_arrival_gen as ag

# The test environment
from lane_arrivals_test_env import LaneArrivalsTestEnv

sim_step = 0.1
horizon = sec2sim(500, sim_step)
m = 1    # Number of rows in grid
n = 3    # Number of columns

simulator = "traci"
sim_params = {
    "restart_instance": False,
    "sim_step": sim_step,
    "print_warnings": True,
    "render": True,
}

env = WolfEnv.create_env(
    cls=LaneArrivalsTestEnv,
    m=m,
    n=n,
    horizon=horizon,
    speed_limit=20,
    simulator=simulator,
    sim_params=sim_params,
)
# Default values of other parameters are recorded in lane_arrivals_test_env
# TODO (Ilia): Currently all vehicles are added with the same initial speed (10)
#              and router/behaviour ('human'). Need to pass a more detailed config
#              to the generator gen_lane_arrivals in flow.envs.multiagent.base

# =================================================================
# =============== Assigning Lane Arrivals by Demand ===============

demand1 = ag.StepFnDemand( (1000, 0.01),
                           (2000, 0.0),
                           (3000, 0.02),
                           (4000, 0.0),
                           (5000, 0.05),
                           horizon=5000)

demand2 = ag.StepFnDemand( (1000, 0.0),
                           (2000, 0.01),
                           (3000, 0.0),
                           (5000, 0.05),
                           horizon=5000)

# Names of lanes can be read off the SUMO-gui
lane1 = 'bot0_0_0'
lane2 = 'left1_0_0'
lane3 = 'right0_0_0'

lane_arrivals = []
lane_arrivals.append( ag.RegularArrivals(lane1, demand1) )
lane_arrivals.append( ag.UniformArrivals(lane2, demand2) )
lane_arrivals.append( ag.NormalArrivals(lane3, demand2) )

# Pool the lane arrivals and attach to the environment
pooled_arrivals = ag.pool_arrivals(*lane_arrivals, horizon=horizon)
env.set_lane_arrival_sched(pooled_arrivals)

traffic_light_ids = ('tl_center' + str(i) for i in range(m*n))
# max agents
traffic_light_actions = dict( (id, EXTEND) for id in traffic_light_ids )

# Run the simulation
env.reset()
for _ in range(horizon):
    env.step(traffic_light_actions)
