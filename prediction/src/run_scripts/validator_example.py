import sys
sys.path.insert(0,'../..')

import torch
import pickle
import numpy as np
import datetime

from src.prediction_models.regression import RegressionPredictor
from src.validator import Validator

with open('../../data/50runs-random.pickle', 'rb') as f:
    aimsun_data = pickle.load(f)
aimsun_data.split(val_pct=0.2, test_pct=0.2)

param_dict = {'model_type' : ['ridge'], 'n_hops' : [2, 3, 4, 5, 6], 'past_steps' : [2, 3, 4, 5, 6], 'horizon': [[5]], 'alpha' : [0, 1e-3, 1e-2, 0.1, 1, 10, 100]}

validator = Validator(RegressionPredictor, aimsun_data, param_dict)

results = validator.cross_validate(k=5, val_pct=0.25)

print('results: ' + str(results))

print(datetime.datetime.now())
