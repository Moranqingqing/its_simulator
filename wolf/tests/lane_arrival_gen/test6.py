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
m = 3    # Number of rows in grid
n = 3    # Number of columns

sim_step = 0.1
impulse_len = 75    # In sim steps
impulse_intensity = 0.05
rounds = 3          # How many times the demand goes around
cooldown = 1800
horizon = 12 * rounds * impulse_len + cooldown

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
    speed_limit=15,
    simulator=simulator,
    sim_params=sim_params,
)
# Default values of other parameters are recorded in lane_arrivals_test_env

# =================================================================
# =============== Assigning Lane Arrivals by Demand ===============

# === TEST 6    Demand impulses going around in a circle

# Names of lanes can be read off the SUMO-gui
# Seems like the map is turned 90 degrees :)
top0 = 'left3_0_0'
top1 = 'left3_1_0'
top2 = 'left3_2_0'
right0 = 'top2_3_0'
right1 = 'top1_3_0'
right2 = 'top0_3_0'
bot0 = 'right0_2_0'
bot1 = 'right0_1_0'
bot2 = 'right0_0_0'
left0 = 'bot0_0_0'
left1 = 'bot1_0_0'
left2 = 'bot2_0_0'

lanes = [top0, top1, top2, right0, right1, right2,
         bot0, bot1, bot2, left0, left1, left2]
lanes = lanes * rounds

demands = []
for i in range(len(lanes)):
    demands.append( ag.StepFnDemand( (impulse_len * i, 0.0),
                                     (impulse_len * (i+1), impulse_intensity),
                                     (horizon, 0.0),
                                     horizon=horizon)
                  )

lane_arrivals = []
for lane, demand in zip(lanes, demands):
    lane_arrivals.append( ag.NormalArrivals(lane, demand, s=0.3) )

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
