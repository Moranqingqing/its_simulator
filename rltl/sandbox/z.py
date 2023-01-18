import numpy as np

r = np.random.RandomState(0)
z0 = r.normal(-1.0, 1.0, size=[10, 2])

r = np.random.RandomState(0)

z1 = []

for i in range(10):
    z1.append(r.normal(-1.0, 1.0, size=[1, 2]))
z1 = np.stack(z1)

print(z0)

print(z1)