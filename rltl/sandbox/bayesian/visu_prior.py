import numpy as np
import matplotlib.pyplot as plt


def p(pi, prior_1, prior_2):
    return pi * np.random.normal(0, prior_1) + (1 - pi) * np.random.normal(0, prior_2)

# for pi in np.arange(0,1,11):
#     for prior_1 in[0.001,0.01,0.1,1,10]:
#         for prior_2 in [0.001, 0.01, 0.1, 1, 10]:
#             x = [p(pi,prior_1,prior_2) for _ in range(1000)]
#
#             plt.hist(x)
#             plt.show()


x = [p(0.5,1.,0.5) for _ in range(100000)]
print(x)
plt.hist(x,bins=1000)
plt.show()