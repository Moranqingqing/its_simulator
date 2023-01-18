import logging

from sow45.utils.configuration.configuration import Configuration
from sow45.world.environments.factories.env_meta_factory import META_FACTORY

C = Configuration().load("configs/ticket24.json").create_fresh_workspace(force=True)

# gym env factory
create_env, gym_name = META_FACTORY.create_factory(**C["env_params"])

test_env = create_env()

loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.FATAL)
# exit()

test_env.reset()
for _ in range(20):
    test_env.step({agent.get_id(): agent.action_space().sample() for agent in test_env.get_agents().values()})

print(logging.getLogger())
