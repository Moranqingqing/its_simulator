"""
Script for testing classes derived from LaneArrivals
(The classes being tested are defined in the lane_arrival_gen module.)
The environment is a grid Traffic Light Control environment based on
GenericGridEnv, but simplified slightly.
"""

from wolf.world.environments.wolfenv.wolf_env import WolfEnv

# Traffic light actions
from wolf.world.environments.wolfenv.agents.connectors.action.exchange_change_phase import EXTEND, CHANGE

from wolf.utils.math import *

# The module being tested
import wolf.world.environments.wolfenv.lane_arrival_gen as ag

# The test environment
from lane_arrivals_test_env import LaneArrivalsTestEnv

sim_step = 0.1
horizon = sec2sim(600, sim_step)
cooldown = 1000
m = 1    # Number of rows in grid
n = 1    # Number of columns

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
    horizon=horizon+cooldown,
    speed_limit=10,
    simulator=simulator,
    sim_params=sim_params,
)
# Default values of other parameters are recorded in lane_arrivals_test_env

# =================================================================
# =============== Assigning Lane Arrivals by Demand ===============

# === TEST 7    Approximating the demand profile given by an arbitrary function
#               by short linear ramps

# Approximating the function
# f(x) = 0.02 * (1 - (4/horizon^2) * (x- horizon/2)^2)
# by linear segments

max_demand = 0.01
delta_s = 10    # Number of simulation steps per segment
demand = ag.DemandProfile(horizon)
num_xs = int(horizon / delta_s) + 1
xs = [ x * delta_s for x in range(num_xs) ]
ys = [ 0.003 + max_demand * ( 1 - (4 / (horizon*horizon)) * (x - horizon/2) * (x - horizon/2) ) for x in xs ]

for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:]):
    demand.add_linear_ramp(x0, x1, y0, y1)
# Plot the demand profile
demand.plot()

# Names of lanes can be read off the SUMO-gui
# The southbound lane
lane = 'left1_0_0'

lane_arrivals = ag.RegularArrivals(lane, demand)

# Pool the lane arrivals and attach to the environment
pooled_arrivals = ag.pool_arrivals(lane_arrivals, horizon=horizon)
env.set_lane_arrival_sched(pooled_arrivals)

traffic_light_ids = ('tl_center' + str(i) for i in range(m*n))
# max agents
traffic_light_actions = dict( (id, EXTEND) for id in traffic_light_ids )

# Run the simulation
env.reset()
for _ in range(horizon + cooldown):
    env.step(traffic_light_actions)
