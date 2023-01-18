'''
Created on Sep. 14, 2020

@author: user
'''

import warnings

from rltl.sandbox.bayesian.DenseVariational import DenseVariational

warnings.filterwarnings('ignore')

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tqdm
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers


# problem
def make_data(n, rots):
    data = []
    for k, rot in enumerate(rots):
        theta = rot * 0.0174533
        rotmat = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
        states = np.random.uniform(size=(n, 2))
        next_states = (rotmat @ states.T).T
        classes = np.ones((n, 1)) * float(k)
        data_k = np.column_stack((states, next_states, classes))
        data.append(data_k)
    data = np.row_stack(data)
    return data


data = make_data(100, [15.0, 30.0])
X = data[:, :-1]
y = data[:, -1:]
y = OneHotEncoder().fit_transform(y)

# model
train_size = y.shape[0]
batch_size = 32
num_batches = train_size / batch_size
prior_params = {'kl_weight' : 1e-2, 'prior_sigma_1': 1., 'prior_sigma_2': 1., 'prior_pi': 0.5 }

x_in = Input(shape=(4,))
x = DenseVariational(40, activation='relu', **prior_params)(x_in)
x = DenseVariational(40, activation='relu', **prior_params)(x)
x = DenseVariational(2, activation='softmax', **prior_params)(x)
model = Model(x_in, x)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.01),
              metrics=['categorical_crossentropy'])

# training
model.fit(X, y, batch_size=batch_size, epochs=500)

# inference
y_pred_list = []
for i in tqdm.tqdm(range(500)):
    y_pred = model.predict(X)[:, 0:1]
    y_pred_list.append(y_pred)
y_preds = np.concatenate(y_pred_list, axis=1)
y_mean = np.mean(y_preds, axis=1)
y_sigma = np.std(y_preds, axis=1)

print(y_mean)
print(y_sigma)
