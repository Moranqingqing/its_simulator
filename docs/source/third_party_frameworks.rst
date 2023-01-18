Third party frameworks
======================

Ray
^^^

Ray is an python API used for multiprocessing (Ray), meta-parameter search (Tune)
and reinforcement learning (RLLib).

They offer a collection of single and multi-agents RL algorithms:

Algorithms and their parameters:
https://docs.ray.io/en/latest/rllib-algorithms.html

We only tried PPO, DQN, APEX-DQN, QMIX and APEX-QMIX with the uoft framework, but others should work.

Make sure to understand how Ray work before diving into WOLF documentation.


Gym
^^^

Gym is a collection of Reinforcement Learning environments. These environments implement the famous Gym interface, where an agent interact
with it through a function called "step". Agent receive observation, reward, information and a done flag after each step.
Gym focuses on single agent formulation. Ray provides a multi-agents formulation that Flow will extend.

Flow
^^^^

Flow is a wrapper around two micro simulators, Aimsum and SUMO. It comes with a gym-like/ray-like environment for Reinforcement Learning.
The environment can simulate multiple agents in either Aimsum or SUMO. The API make is easier to create network programatically
(like the GridEnv network), or load open street maps and simulate "real" networks. Primarily focus of flow is autonomous driving and
the support for traffic light control is very limited. This is why provide a wrapper around Flow environment. Unlike Flow,
our environment support all kind of components like controllers, rewards and observations,
all compatible with a single container class, TrafficEnv.

