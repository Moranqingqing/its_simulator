Introduction
============

WOLF (What to do to Optimise Large Flows) is a python API developed by the University of Toronto, in collaboration with Huawei Canada. Its aim is
to offer Traffic Light Control optimisation on micro (Aimsum, SUMO) and macro (Cellular Transmission Model) simulators. It relies on Berkley's Flow API
for traffic micro simulation (a wrapper around Aimsim and SUMO), and the Ray framework for multiprocessing and RL optimisation.

The main contribution of WOLF is a wrapper around Flow RL environment. Our wrapper allow a lot of customisation
and more tool for traffic light optimisation (Flow lacks of those features). This wrapper is very
similar to a gym environment. You can learn a single-agent or multi-agents policy to interact with this
environment. WOLF uses RLLib (a sub-API of Ray) to optimise those policies, but you are free to use something
else (openAI baselines, stable-baseline, dopamine, TF-agent etc), however you will lack some features we offer
(visualisation scripts and plotting results). See :ref:`this section<functional_api>` for a functional use of our API.

In order to understand most of the code and configuration of WOLF, one needs to understand how Flow and Ray works.