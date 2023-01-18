## Tests for classes derived from LaneArrivals
(RegularArrivals, BernoulliArrivals, PoissonArrivals, UniformArrivals, NormalArrivals)

These classes, together with the LaneArrivals interface class, are all defined
in the module lane_arrival_gen.

All of the test environments are grids.


__Test 1 (Five variants)__

Three southbound lanes, with different but constant demand rates, and
the same type of demand on each lane (Regular, Bernoulli, etc.).
Traffic lights are green in all directions.


__Test 2 (Three variants)__

Five roads, with the same constant demand rate, but different types of demand
Left-to-right, the types of demand are
    Regular -> Bernoulli -> Poisson -> Uniform -> Normal
The traffic lights are always green in all directions


__Test 3 (Three variants)__

Similar to Test 2, but the traffic lights are *red* in all directions.
Comparing how quickly the lanes fill up is a way of making sure that
the arrival rates are approximately equal for the different arrival types.


__Test 4__

Similar to Tests 2 and 3, but with a linear ramp demand profile.


__Test 5__

Similar to Tests 2, 3 and 4, but with a triangular wave demand profile.


__Test 6__

A 3x3 grid, with demand impulses going around in a circle on the outer edges.
The traffic lights are always green in all directions.

__Test 7__

Linear ramps may seem restrictive, but you can approximate many functional forms
very well by using many short linear ramps.

In this test, a parabolic demand profile is created using short linear ramps,
and tested on a single intersection.


__Test 8__

Similar to Test 1 - Bernoulli, but with environment resets


__Test 9__

Similar to Test 1 - Bernoulli, but with lane arrivals attached to the
environment at initialization instead of post-initialization.
