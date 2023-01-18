.. _config_file_example:

Config file example
===================

Here is an example :code:`.yaml` file for running an APEX-DQN experiment in grid (synthetic) network environment with the following settings:

* State space: TDTSE (shor-range detection)
* Action space: extend-change, no progression, no masking
* Reward function: queue length

In this example file, we commented on most of the important arguments. For more details about ray's arguments, please refer to labeled ray's relative documentations. For more details about wolf's arguments, please refer to the corresponding code documentation.

**Notice:** label :code:`# REG <module name>` in some lines represents you can replace that module with any others shown in :code:`wolf.utils.configuration.registry.py`. If you want to replace them, remember to add necessary parameters for the new module. Full describtion of parameters can be found in corresponding files.


This is an example of a configuration file.

.. code-block:: yaml

    # arguments related to the experiment, including settings of ray, algorithm, environment, etc.
    "ray":
      # ray.init(), refer to: https://docs.ray.io/en/master/package-ref.html?highlight=init#ray.init
      "init":
        "local_mode": false
        "log_to_driver": true
        "logging_level": "WARNING"
      "run_experiments":
        "experiments":
          "global_agent":    # name of the experiment

            ####################
            # Trainable algorithm: APEX/DQN/EVALUATOR/...
            # Refer to: https://docs.ray.io/en/master/tune/api_docs/execution.html?highlight=tune.experiments#ray.tune.Experiment
            ####################
            "run": "APEX"
            "checkpoint_freq": 1
            "checkpoint_at_end": true
            "stop":
              "training_iteration": 100
            
            ####################
            # APEX configuration
            # Refer to: https://docs.ray.io/en/master/rllib-algorithms.html?highlight=algorithms#rllib-algorithms
            # APEX-DQN section
            ####################
            "config":

              ####################
              # OTHERS
              ####################
              "framework": "tf"
              "log_level": "WARNING"

              ####################
              # RL ALGO PARAMS
              ####################
              "num_gpus": 1
              "num_workers": 7    # at least 2 for APEX
              "target_network_update_freq": 100
              "learning_starts": 0
              "timesteps_per_iteration": 1000

              ####################
              # EVALUATION
              ####################
              # "evaluation_interval": 10
              # "evaluation_num_episodes": 10
              # "in_evaluation": False
              # "evaluation_config":
              #   "explore": False
              # "evaluation_num_workers": 0
              # "custom_eval_function": null
              # "use_exec_api": False

              ####################
              # EXPLORATION
              ####################
              # "exploration_config":
              #   "type": "EpsilonGreedy"
              #   "epsilon_schedule":
              #     "type": "ExponentialSchedule"
              #     "schedule_timesteps": 10000
              #     "initial_p": 1.0
              #     "decay_rate": 0.01

              ####################
              # CUSTOM MODEL
              ####################
              "model":
                # identify the custom model here. Replaceable models: refer to
                # wolf.utils.configuration.configuration.py load_custom_models method
                "custom_model": "tdtse"
                "custom_model_config":
                  # number of kernels in each CNN layer
                  "filters_size": 32
                  # size of the final FNN layer
                  "dense_layer_size_by_node": 64
                  # whether to use progression, this value should be consist with the
                  # "use_progression" in "action_params"
                  "use_progression": false

              ####################
              # ENVIRONMENT
              ####################
              "gamma": 0.99
              # simulation horizon, if null, horizon will be choosen by env
              "horizon": null
              # environment instance
              "env": "traffic_env_test0"    # REG env_factory
              # environment configuration
              "env_config":
                # the simulator will be used
                "simulator": "traci"
                # simulation related parameters
                "sim_params":
                  # whether to restart a simulation upon reset
                  "restart_instance": True
                  # simulation time step length (unit: s)
                  "sim_step": 1
                  # whether to print simulation warnings
                  "print_warnings": False
                  # whether to run with render (SUMO-GUI in this case)
                  "render": False
                "env_state_params": null
                "group_agents_params": null
                # identify how to build the policy mapping
                "multi_agent_config_params":
                  # multiple agents share weights or not
                  "name": "shared_policy"    # REG multi_agent_config_factory
                  "params": {}
                # agent related parameters
                "agents_params":
                  # agent type: global or independent
                  "name": "global_agent"    # REG agent_factory
                  "params":
                    # if running an EVALUATOR, identify the policy here
                    "default_policy": null    # REG policy
                    # whether to use global reward
                    "global_reward": false
                    
                    ####################
                    # CONNECTORS
                    # refer to: wolf.environment.traffic.agents.connectors
                    ####################

                    ### ACTION SPACE
                    "action_params":
                      "name": "ExtendChangePhaseConnector"    # REG connector
                      "params": {}

                    ### OBSERVATION SPACE
                    "obs_params":
                      "name": "TDTSEConnector"    # REG connector
                      "params":
                        "obs_params":
                          "num_history": 60
                          "detector_position": [5, 100]
                        "phase_channel": true

                    ### REWARD FUNCTION
                    "reward_params":
                      "name": "QueueRewardConnector"    # REG connector
                      "params":
                        "stop_speed": 2

    # general arguments
    "general":
      "id": "main"
      "seed": null
      "repeat": 1
      "is_tensorboardX": false
      
      # if SUMO_HOME is not in your system path,
      # please identify it here
      # "sumo_home": "/home/ncarrara/sumo_binaries/bin"
      
      # output file path
      "workspace": "wolf/tests/traffic_env/test0/results"

      "logging":
        "version": 1
        "disable_existing_loggers": false
        "formatters":
          "standard":
            "format": "[%(name)s] %(levelname)s - %(message)s"
        "handlers":
          "default":
            "level": "WARNING"
            "formatter": "standard"
            "class": "logging.StreamHandler"
        "loggers":
          "":
            "handlers": ["default"]
            "level": "WARNING"
            "propagate": false
          "some.logger.you.want.to.enable.in.the.code":
            "handlers": ["default"]
            "level": "ERROR"
            "propagate": false


