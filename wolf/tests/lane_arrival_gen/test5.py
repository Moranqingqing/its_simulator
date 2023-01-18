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

# Test parameters
m = 1    # Number of rows in grid
n = 5    # Number of columns
sim_step = 0.1
demand_period = 3000
demand_min = 0.003
demand_max = 0.025
num_periods = 2
cooldown = 100
horizon = int(demand_period * (num_periods + 0.5) + cooldown)

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
    speed_limit=10,
    simulator=simulator,
    sim_params=sim_params,
)
# Default values of other parameters are recorded in lane_arrivals_test_env

# =================================================================
# =============== Assigning Lane Arrivals by Demand ===============

# === TEST 5   Five roads with a triangular wave demand, and different types

peaks_and_troughs = []
step = 0
for p in range(num_periods):
    peaks_and_troughs.append( (step, demand_min) )
    peaks_and_troughs.append( (step + (demand_period / 2), demand_max) )
    step += demand_period
peaks_and_troughs.append( (step, demand_min) )

demand = ag.PiecewiseLinearDemand(*peaks_and_troughs, horizon=horizon)

# Names of lanes can be read off the SUMO-gui
# The southbound lanes
lane0 = 'left1_0_0'
lane1 = 'left1_1_0'
lane2 = 'left1_2_0'
lane3 = 'left1_3_0'
lane4 = 'left1_4_0'

lane_arrivals = []
lane_arrivals.append( ag.RegularArrivals(lane0, demand) )
lane_arrivals.append( ag.BernoulliArrivals(lane1, demand) )
lane_arrivals.append( ag.PoissonArrivals(lane2, demand) )
lane_arrivals.append( ag.UniformArrivals(lane3, demand) )
lane_arrivals.append( ag.NormalArrivals(lane4, demand) )

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
