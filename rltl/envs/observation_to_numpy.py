import numpy as np

def obs_to_np_factory(type: str):
    if type == "identity":
        return lambda obs: obs

    if type == "queue_length":
        return lambda obs: np.array([obs["queue"]])
