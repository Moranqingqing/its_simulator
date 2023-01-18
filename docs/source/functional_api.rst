.. _functional_api:

Functional API
==============

WOLF offer a workflow based on configuration file, run by a single script, main.py. If you need more freedom when designing
your WOLF experiment, you can use TrafficEnv as a classic Gym/RLlib environment.

You can instanciate directly TrafficEnv by discarding two parameters: multi_agent_config_params and group_agents_params. Those arguments are
only used in Ray (and Ray's Qmix).

The functional API has not been maintained since a couple now, but you can see an example in wolf/tests/misc/gaussian_flow.py.
In this example, we instanciate the subclass "SimpleGridEnv" which is a grid-shape network. To interact with it, the agent feed a
dictionary of action (one by agent) using the step function:

.. code-block:: python

    env.step({"tl_center0": EXTEND})


This instruction means there is only one traffic_light, executing action "EXTEND". The return of this step function follows the same
structure as the step function of a multi-agents  `RLLib's environment <https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical>`_.

