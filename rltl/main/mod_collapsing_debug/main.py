import numpy as np
import matplotlib.pyplot as plt

def rotation(x, y, theta_mu, theta_sigma):
    theta = np.random.normal(theta_mu, theta_sigma, 1)[0]
    xp = x + np.cos(theta)
    yp = y + np.sin(theta)
    return xp, yp


N = 1000
sigmas = [0.1, 0.05, 0.25, 1.25]
samples = [[] for _ in range(4)]
for i in range(4):
    c = np.zeros(4)
    c[i] = 1
    for _ in range(N):
        x, y = np.random.random((2,))

        idx = 0
        if x < 1 and y < 1:
            idx = 0
        elif x >= 1 and y >= 1:
            idx = 3
        elif x < 1 and y >= 1:
            idx = 2
        else:
            idx = 1

        xp, yp = rotation(x, y, 0, sigmas[idx + i % 4])
        sample = {"c": c, "s": [x, y], "s_": [xp, yp]}
        samples[i].append(sample)

for sample in samples[0]:
    print(sample)


