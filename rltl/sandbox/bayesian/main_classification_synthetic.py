'''
Created on Sep. 14, 2020

@author: user
'''

import warnings
warnings.filterwarnings('ignore')

import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers


# problem
def gen_data(pi, mus, Sigmas, N):
    classes = np.sort(np.random.choice(len(pi), p=pi, size=(N,)))
    samples = []
    for K in range(len(pi)):
        n_K = (classes == K).sum()
        samples.append(np.random.multivariate_normal(mus[K], Sigmas[K], size=(n_K,)))
    samples = np.concatenate(samples, axis=0)
    return samples, classes


def plot_scatter(samples, classes, name=''): 
    K = np.unique(classes).size
    _, ax = plt.subplots(figsize=(8, 6))
    for k in range(K):
        xyk = samples[classes == k]
        ax.scatter(xyk[:, 0], xyk[:, 1], label='class {}'.format(k))
    plt.legend()
    plt.savefig(name)
    plt.show()


pi_close = [0.5, 0.5]
means_close = [[-0.25, -0.25], [0.25, 0.25]]
sigmas_close = [np.eye(2) * 0.15, np.eye(2) * 0.12]

pi_far = [0.5, 0.5]
means_far = [[-0.75, -0.75], [0.75, 0.75]]
sigmas_far = [np.eye(2) * 0.1, np.eye(2) * 0.075]

if False:
    pi, means, sigmas, label = pi_close, means_close, sigmas_close, 'close'
else:
    pi, means, sigmas, label = pi_far, means_far, sigmas_far, 'far'

samples, classes = gen_data(pi, means, sigmas, 1000)
plot_scatter(samples, classes, '{}_data.png'.format(label))
enc = OneHotEncoder()
enc.fit(classes.reshape((-1, 1)))
y = enc.transform(classes.reshape((-1, 1))).toarray()

# model
train_size = samples.shape[0]
batch_size = 32
num_batches = train_size / batch_size
kl_weight = 1.0 / num_batches
prior_params = { 'prior_sigma_1': 1.0, 'prior_sigma_2': 0.1, 'prior_pi': 0.5 }
 
x_in = Input(shape=(2,))
x = DenseVariational(50, kl_weight, **prior_params, activation='relu')(x_in)
x = DenseVariational(50, kl_weight, **prior_params, activation='relu')(x)
x = DenseVariational(2, kl_weight, **prior_params, activation='softmax')(x)
model = Model(x_in, x)
model.compile(loss=DenseVariational.cross_entropy_negative_log_likelihood(),
              optimizer=optimizers.Adam(lr=0.05),
              metrics=['categorical_crossentropy'])
 
# training
model.fit(samples, y, batch_size=batch_size, epochs=200)

 
# inference
def plot_boundary(xx, yy, samples, classes, probs, name): 
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 100, cmap="RdBu", vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    K = np.unique(classes).size
    for k in range(K):
        xyk = samples[classes == k]
        ax.scatter(xyk[:, 0], xyk[:, 1], s=6, label='class {}'.format(k))
    plt.legend()
    plt.savefig(name)
    plt.show()


samples, classes = gen_data(pi, means, sigmas, 500)
xmin, xmax = -2., 2.
ymin, ymax = -2., 2.
xstp, ystp = 0.05, 0.05
xx, yy = np.mgrid[xmin:xmax:xstp, ymin:ymax:ystp]
grid = np.c_[xx.ravel(), yy.ravel()]
print(grid.shape)
y_pred_list = []
for i in tqdm.tqdm(range(200)):
    y_pred = model.predict(grid)
    y_pred_list.append(y_pred)
y_preds = np.stack(y_pred_list, axis=2)
y_mean = np.mean(y_preds, axis=2)[:, 1].reshape(xx.shape)
y_sigma = np.std(y_preds, axis=2)[:, 1].reshape(xx.shape)
plot_boundary(xx, yy, samples, classes, y_mean, '{}_prediction_mean.png'.format(label))
plot_boundary(xx, yy, samples, classes, y_sigma, '{}_prediction_var.png'.format(label))
