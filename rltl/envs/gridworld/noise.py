import numpy as np


def dynamics(x, y, ax, ay, dynamics_params):
    dynamics_type = dynamics_params["type"]
    lambda_ = dynamics_params["lambda"]
    if dynamics_type == "translation":
        xt, yt = lambda_(x, y)
        dx, dy = ax + xt, ay + yt
        xp, yp = x + dx, y + dy
    elif dynamics_type == "rotation":
        theta_mu, theta_sigma = lambda_(x, y)
        theta = np.random.normal(theta_mu, theta_sigma, 1)[0]
        xp = x + ax * np.cos(theta) - ay * np.sin(theta)
        yp = y + ax * np.sin(theta) + ay * np.cos(theta)
    elif dynamics_type == "discrete_rotation":
        theta_mu, theta_sigma = lambda_(x, y)
        theta = np.random.normal(theta_mu, theta_sigma, 1)[0]
        rot90 = np.pi / 2
        theta = int(round(theta / rot90)) * rot90
        xp = x + ax * np.cos(theta) - ay * np.sin(theta)
        yp = y + ax * np.sin(theta) + ay * np.cos(theta)
    elif dynamics_type == "s+a":
        xp = x + ax
        yp = y + ay
    else:

        raise Exception("Unknown dynamics type: {}".format(dynamics_type))
    return xp, yp
