import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from operator import itemgetter
from abc import ABCMeta, abstractmethod
from wolf.utils.math import EPSILON

# =============================================================================
# =========================== Demand Profile Class ============================

class DemandProfile():
    """
    A class for describing the rate of traffic generation (demand) on a
    single lane, over the course of a simulation. Intended to be passed
    to a class derived from LaneArrivals (ex. BernoulliArrivals), which
    will generate a list of vehicle arrivals on the lane according to the
    demand profile.

    The demand profile is represented by a function that sends the index of
    a simulation step (the index lies in the set 0, 1, ..., horizon-1) to
    the rate of vehicle generation at that simulation step (measured in
    vehicles per simulation step, which is a float).

    There are two intended ways of creating a demand profile:

    1. API-like. An example of intended usage is

     >>> demand = DemandProfile(30)                  # Initialize the class with a horizon of 30 steps
     >>> demand.add_constant_section(0, 15, 0.3)     # Start with constant demand of 0.3 for 15 steps
     >>> demand.add_linear_ramp(15, 30, 0.3, 0.5)    # Ramp up demand from 0.3 to 0.5 over the next 15 steps
     >>> demand.plot()                               # Convenience function to plot the graph of the profile

    2. Creating a derived class with a faster initialization (for examples,
       please see the classes LinearRampProfile and PiecewiseLinearProfile)
    """

    def __init__(self, horizon):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.rates = [0] * self.horizon

    def add_linear_ramp(self, step_i, step_f, rate_i, rate_f):
        """
        Adds a section of the demand profile that linearly ramps from rate rate_i
        to rate_f, over the simulation interval step_i <= step < step_f
        """
        step_i = int(step_i)
        step_f = int(step_f)
        if step_f > self.horizon:
            raise ValueError(
                  f'Parameter step_f ({step_f}) lies beyond the simulation '
                  f'horizon ({self.horizon})'
            )

        if step_f < step_i:
            raise ValueError(
                  f'Parameter step_f ({step_f}) should be greater than '
                  f'parameter step_i ({step_i})'
            )

        if step_f == step_i:
            return

        if step_i < 0:
            raise ValueError(f'Parameter step_i ({step_i}) should be a positive')

        if not ((0 <= rate_i <= 1) and (0 <= rate_f <= 1)):
            raise ValueError(
                  f'Vehicle generation rates (per simulation step) must lie between '
                  f'0 and 1 inclusive (obtained rate_i = {rate_i} and rate_f = {rate_f})'
            )

        slope = (rate_f - rate_i) / (step_f - step_i)
        ramp = [ rate_i + slope * t for t in range(step_f - step_i) ]

        self.rates[step_i : step_f] = ramp

    def add_constant_section(self, step_i, step_f, rate):
        """
        Adds a section of the demand profile with constant demand of the
        given rate, over the simulation interval step_i <= step < step_f
        """
        self.add_linear_ramp(step_i, step_f, rate, rate)

    def add_certain_arrival(self, step):
        """
        Adds a certain arrival at the specified simulation step (use with
        Bernoulli)
        """
        self.add_constant_section(step, step+1, 1)

    def plot(self):
        """ Plots the graph of the demand profile """
        sim_steps = tuple(range(self.horizon))
        plt.plot(sim_steps, self.rates)
        plt.xlabel('Simulation step')
        plt.ylabel('Rate of demand')
        plt.show()

    def __str__(self):
        """ Prints out the values of the demand function as a string (rounded to 3 decimal places) """
        return str([round(rate, 3) for rate in self.rates])


# Derived classes for common demand profiles, with a quicker initialization

class ConstantDemand(DemandProfile):
    """ Creates a DemandProfile with a constant demand rate
        over the entire simulation """
    def __init__(self, horizon, rate):
        super().__init__(horizon)
        self.add_constant_section(0, horizon, rate)

class StepFnDemand(DemandProfile):
    """
    Takes a collection of 2-tuples (step, rate), and generates a DemandProfile
    that is a step function.

    For example, if (14, 0.2) and (19, 0.3) are two consecutive tuples, then
    the demand profile takes the value 0.3 in the interval 14 <= step < 19
    (the value 0.2 is used for the previous profile section).
    (For the first 2-tuple, the previous point is at sim_step=0)
    """
    def __init__(self, *tuples, horizon):
        super().__init__(horizon)

        if tuples is not None:
            sorted(tuples, key=itemgetter(0))    # Sort by simulation step index
            prev_step = 0
            for step, rate in tuples:
                self.add_constant_section(prev_step, step, rate)
                prev_step = step

class LinearRampDemand(DemandProfile):
    """ Creates a DemandProfile that linearly ramps from rate_i
        to rate_f over the course of the simulation """
    def __init__(self, horizon, rate_i, rate_f):
        super().__init__(horizon)
        self.add_linear_ramp(0, horizon, rate_i, rate_f)

class PiecewiseLinearDemand(DemandProfile):
    """
    Takes a collection of 2-tuples (step, rate), and generates a DemandProfile
    that connects consecutive 2-tuples by linear segments

    For example, the following recovers the same demand profile as the example in the
    description of the DemandProfile class:

      >>> demand = PiecewiseLinearDemand((0, 0.3), (15, 0.3), (30, 0.5), horizon=30)
    """
    def __init__(self, *tuples, horizon):
        if len(tuples) < 2:
            raise ValueError('Expect at least two 2-tuples to be passed to '
                             'PiecewiseLinearDemand')
        super().__init__(horizon)

        sorted(tuples, key=itemgetter(0))    # Sort by simulation step index
        cur_step, cur_rate = tuples[0]
        for next_step, next_rate in tuples[1:]:
            self.add_linear_ramp(cur_step, next_step, cur_rate, next_rate)
            cur_step, cur_rate = next_step, next_rate


# =============================================================================
# ============================== Lane Arrivals ================================

DEFAULT_SEED = 1234
DEFAULT_TRESHOLD = 0.001

class LaneArrivals(metaclass=ABCMeta):
    """
    Abstract class that provides a common interface for classes that generate
    vehicle arrival times on a fixed lane, according to the given demand profile.

     Parameters
     ----------
       lane_id : string
           The Wolf id of the lane on which the vehicles are generated
       demand_profile : DemandProfile
           Class describing the demand on the specified lane
       seed : int, optional
           Seed for the random number generator
       TRESHOLD : float, optional
           Rates below the TRESHOLD are ignored, otherwise the waiting times are
           too long and an increase in demand may be missed

     Attributes
     ----------
     Arrivals are represented by 2-tuples (step, veh_id) of type (int, string),
     where step is the simulation step of the arrival of the vehicle, and veh_id
     is the assigned vehicle id

     Arrivals is an iterable that can be accessed by index. It is represented by
     a list, but does not have to be. It is built on initialization.

     A waiting time is the time between two vehicle arrivals.
    """
    @abstractmethod
    def __init__(self, lane_id, demand_profile,
                 seed=DEFAULT_SEED, TRESHOLD=DEFAULT_TRESHOLD):
        self.seed = seed
        self.TRESHOLD = TRESHOLD
        self.lane_id = lane_id
        self.demand = demand_profile
        self.arrivals = []
        self.gen()
        self.num_arrivals = len(self.arrivals)
        self.i = 0

    def assign_veh_id(self, veh_num):
        """ Generates a vehicle identifier based on an index """
        return self.lane_id + '.v' + str(veh_num)

    @abstractmethod
    def gen(self):
        """ Generates the list of arrivals """
        pass

    # Abstracting away the representation of self.arrivals as a list...
    def __len__(self):
        return self.num_arrivals

    def __getitem__(self, key):
        if 0 <= key < self.num_arrivals:
            return self.arrivals[key]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.num_arrivals:
            val = self.arrivals[self.i]
            self.i += 1
            return val
        else:
            raise StopIteration

    def __str__(self):
        return str(self.arrivals)

    def append(self, item):
        self.arrivals.append(item)


class RegularArrivals(LaneArrivals):
    """
    Generates a sequence of equally-spaced (as much as possible) vehicles
    on the lane lane_id, at the rates described by the demand_profile
    """
    def __init__(self, lane_id, demand_profile,
                 TRESHOLD=DEFAULT_TRESHOLD):
        super().__init__(lane_id, demand_profile, TRESHOLD=TRESHOLD)

    def gen(self):
        step = 1
        while step < self.demand.horizon:
            rate = self.demand.rates[step]
            if rate < self.TRESHOLD:        # If the rate is too low, ignore it, otherwise
                step += 1                   # may wait too long for the next car and miss
                continue                    # an increase in demand
            veh_id = self.assign_veh_id(len(self.arrivals))
            self.append( (step, veh_id) )
            gap = 1/(rate + EPSILON)
            step += gap
            step = ceil(step)


class BernoulliArrivals(LaneArrivals):
    """
    Flips a coin every simulation step to decide whether or not to generate
    a vehicle on lane lane_id. The probability of success is read off the
    demand_profile as the rate
    """
    def __init__(self, lane_id, demand_profile,
                 seed=DEFAULT_SEED, TRESHOLD=DEFAULT_TRESHOLD):
        super().__init__(lane_id, demand_profile, seed, TRESHOLD)

    def gen(self):
        for step in range(1, self.demand.horizon):
            rate = self.demand.rates[step]
            if np.random.binomial(1, rate):
                veh_id = self.assign_veh_id(len(self.arrivals))
                self.append( (step, veh_id) )


class PoissonArrivals(LaneArrivals):
    """
    Generates vehicles by a (discretized) Poisson process at the lane lane_id,
    with the rate of incoming vehicles read off the demand_profile

    Note: Because time is discretized in the simulation, this produces only an
          approximation of the Poisson process, and is essentially the same as
          BernoulliArrivals. As there are fewer 'coin flips', PoissonArrivals
          tends to be about 10-20 times faster, so it is kept.
    """
    def __init__(self, lane_id, demand_profile,
                 seed=DEFAULT_SEED, TRESHOLD=DEFAULT_TRESHOLD):
        super().__init__(lane_id, demand_profile, seed, TRESHOLD)

    def gen(self):
        step = 1
        while step < self.demand.horizon:
            rate = self.demand.rates[step]
            if rate < self.TRESHOLD:
                step += 1
                continue
            veh_id = self.assign_veh_id(len(self.arrivals))
            self.append( (step, veh_id) )
            scale = 1/(rate + EPSILON)    # For use with the exp. distribution in numpy
            step += np.random.exponential(scale)
            step = ceil(step)


class UniformArrivals(LaneArrivals):
    """
    Generates vehicles with waiting times uniformly distributed in the interval
    [gap - gap * mult, gap + gap * mult], where the variables are :
        gap = 1/rate   : float, read off the demand_profile
        0 <= mult <= 1 : float, optionally passed at init, otherwise 1/2
    """
    def __init__(self, lane_id, demand_profile, mult=1/2,
                 seed=DEFAULT_SEED, TRESHOLD=DEFAULT_TRESHOLD):
        if not (0 <= mult <= 1):
            raise ValueError(f'Spread multiplier mult should lie '
                             f'between 0 and 1. Received mult={mult}')

        self.mult = mult
        super().__init__(lane_id, demand_profile, seed, TRESHOLD)

    def gen(self):
        step = 1
        while step < self.demand.horizon:
            rate = self.demand.rates[step]
            if rate < self.TRESHOLD:
                step += 1
                continue
            veh_id = self.assign_veh_id(len(self.arrivals))
            self.append( (step, veh_id) )
            gap = 1/(rate + EPSILON)
            step += np.random.uniform(  low=gap * (1 - self.mult),
                                       high=gap * (1 + self.mult) )
            step = ceil(step)


class NormalArrivals(LaneArrivals):
    """
    Generates vehicles with waiting times normally distributed with
    mean=gap and var=(gap*s)^2, where the variables are :
        gap = 1/rate   : float, read off the demand_profile
        0 <= s <= 1/2  : float, optionally passed at init, otherwise 0.25
    The result is clipped at (gap +/- 2*s)
    """
    def __init__(self, lane_id, demand_profile, s=0.25,
                 seed=DEFAULT_SEED, TRESHOLD=DEFAULT_TRESHOLD):
        if not (0 <= s <= 1/2):
            raise ValueError(f'St.dev. multiplier s should lie '
                             f'between 0 and 1/2. Received s={s}')

        self.s = s
        super().__init__(lane_id, demand_profile, seed, TRESHOLD)

    def gen(self):
        step = 1
        while step < self.demand.horizon:
            rate = self.demand.rates[step]
            if rate < self.TRESHOLD:
                step += 1
                continue
            veh_id = self.assign_veh_id(len(self.arrivals))
            self.append( (step, veh_id) )
            gap = 1/(rate + EPSILON)
            # Sample normally distributed gap multiplier
            mult = np.random.normal( loc=1, scale=self.s )
            # Clip the result to lie within two st.dev.
            mult = np.clip(mult, 1-2*self.s, 1+2*self.s)
            step += gap * mult
            step = ceil(step)


def pool_arrivals(*lane_arrivals, horizon):
    """
    Pools a collection of LaneArrivals into a list of lanes that generate
    a vehicle, indexed by timestep. This format is convenient for being
    passed to an environment that will generate the vehicles.

    Parameters
    ----------
        lane_arrivals : collection of LaneArrivals
            Collection of vehicle lane arrivals
        horizon : int
            Number of steps in the simulation

    Return
    ------
        active_lanes_by_step : tuple of lists of 2-tuples (string, string)
            A tuple of lanes that have a vehicle arrival, organized by
            simulation step

            For each index of the tuple, the corresponding list entry
            contains the data (lane_id, veh_id)
                lane_id : str
                    Name of generating lane
                veh_id : str
                    Name of vehicle

                ex. ( [('lane1', 'veh1'), ('lane2', 'veh2')],
                      [('lane3', 'veh3')],
                      ...
                    )
                Represents that at step 0, veh1 arrives at lane1 and veh2 at lane2,
                                at step 1, veh3 arrives at lane3,
                      ...
    """
    active_lanes_by_step = [ [] for _ in range(horizon) ]

    # Need to keep track of the number of times a lane is used to
    # assign unique vehicle identifiers
    lane_count_dict = dict()

    for lane_data in lane_arrivals:
        lane_id = lane_data.lane_id
        lane_count = (lane_count_dict[lane_id] if lane_id in lane_count_dict else 0) + 1
        lane_count_dict[lane_id] = lane_count

        for step, veh_id in lane_data:
            veh_id += '.' + str(lane_count)
            active_lanes_by_step[step].append( (lane_id, veh_id) )
    return tuple(active_lanes_by_step)
