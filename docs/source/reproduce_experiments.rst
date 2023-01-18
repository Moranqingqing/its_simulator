.. _reproduce_experiments:

Reproduce Experiments
=====================

CTM
^^^

For all tests:

.. code-block:: sh

    python main.py ../tests/ctm/test$1/global_agent.yaml

Where $1 in [0,0_1,2,3,4,5]. For test 2 3 and 4, you can use random_agent.yaml instead.

TrafficEnv
^^^^^^^^^^

Synthetic Network (Grid-Net)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For all tests:

.. code-block:: sh

    python main.py ../tests/traffic_env/test$1/$2.yaml

Where $1 in [0,0_1,2] and $2 in [global_agent, random, static_min].

Test0 is a single intersection with demand coming from one direction.

Test0_1 is like Test0, but loopdetectors are more upstream, and the demand is platton like.

Test1 has 2 intersections with a uniform inflow, with a master slave configuration, ie on the direction has a lot of demand, while the
demand on others directions is very low.

Test2 is like test1 but with 4 intersections.

Real-World Network
~~~~~~~~~~~~~~~~~~

Real_net is a environment handling traffic network and demand given by specific files.

You can run a single agent DQN (APEX) on wujiang network (single main intersection) with this command:

.. code-block:: sh

    python main.py ../tests/traffic_env/real_net/dqn.yaml

Or run a multi-agent IQL on shenzhen network (3x3 main intersections) with this command:

.. code-block:: sh

    python main.py ../tests/traffic_env/real_net/iql.yaml

QMIX
^^^^

Until very recently, Ray's QMIX was bugged (it did too much exploration). So we had to work on the original QMIX implementation.
So for the moment, testing QMIX follows another pattern:

QMIX has been shifted to the flow directory. 
From the flow directory, follow the below commands:

.. code-block:: sh

    python qmix/src/main.py --config=qmix_traf --env-config=traf

Here the configuration has been broken into configuration for the algorithm (:code:`--config`) and configuration for the environment (:code:`--env-config`).

Within :code:`--env-config` we specify :code:`traf` which makes the program read the :code:`traf.yaml` inside the :code:`qmix/src/config/envs/` directory.

From this file one can change the:

- :code:`test_env` to change between different wolf registered environments.

- :code:`render` to render the simulation or not.

And other environment specific configuration properties.
