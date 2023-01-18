from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.schedules import Schedule


def show_schedule(config):


    schedule = from_config(
        Schedule,
        config,
        framework=None)

    y = [schedule._value(t) for t in range(config["schedule_timesteps"])]

    import matplotlib.pyplot as plt

    plt.plot(range(config["schedule_timesteps"]), y)
    plt.show()

if __name__=="__main__":
    config = dict(
        type="ExponentialSchedule",
        schedule_timesteps=200000,
        framework=None,
        initial_p=1.0,
        decay_rate=0.001)
    show_schedule(config)
