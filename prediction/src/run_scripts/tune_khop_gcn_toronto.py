import sys
import time
import datetime
import pickle
sys.path.insert(0, r'F:\PhD\OneDrive - University of Toronto\PhD\Projects\sow45_code\prediction')
sys.path.insert(0, '/home/bruceli/sow45_code/prediction')

from src.metrics import *
from src.validator import Validator
from src.prediction_models.neural_nets.gcn.khop_gcn import GCNPredictor
import argparse

args = argparse.ArgumentParser(description='arguments')
args.add_argument('--ps', default=[9], type=eval, help='past_steps')
args.add_argument('--hor', default=[[5]], type=eval, help='horizon')
args.add_argument('--path', type=str, help='path to traffic file. e.g. /home/bruceli/data/toronto/qew/50runs-random.pickle')
args.add_argument('--dataname', type=str, help='for saving model name: qew, dt, pems, whatever.')

args = args.parse_args()
past_steps = args.ps
horizon = args.hor
traffic_data_path = args.path
dataname = args.dataname
print(past_steps, type(past_steps))
print(horizon, type(horizon))
print(dataname)
print(traffic_data_path)
start = time.time()

with open(traffic_data_path, 'rb') as f:
    traffic_data = pickle.load(f)

traffic_data.split(val_pct=0.2, test_pct=0.2)
traffic_data.set_normalize('standard')
traffic_data.transform()

param_dict = {'past_steps' : past_steps, 'hidden_size' : [32, 64, 128], 'batch_size': [64, 32 , 128],
              'learning_rate' : [1e-3, 1e-4, 1e-5], 'weight_decay' : [1e-3, 1e-4, 1e-5],
              'horizon': horizon, 'max_epoch' : [128], 'source_feature':[['flow']],
              'target_feature':[['flow']], 'is_pems':[False]}


validator = Validator(GCNPredictor, traffic_data, param_dict)
params, model = validator.tune_model(return_best_model=True)
print('best params: ' + str(params))

mae_errors = {'train':[], 'val':[], 'test':[]}
mse_errors = {'train':[], 'val':[], 'test':[]}
mape_errors = {'train':[], 'val':[], 'test':[]}

def compute_errors(true, predicted):
    metric = MAEMetrics(true, predicted)
    mae_error = metric.evaluate()

    metric = MSEMetrics(true, predicted)
    mse_error = metric.evaluate()

    metric = MAPEMetrics(true, predicted)
    mape_error = 100.*metric.evaluate() # hundred percent
    return mae_error, mse_error, mape_error

true, predicted = model.evaluate('train')
mae_error_train, mse_error_train, mape_error_train = compute_errors(true, predicted)
mae_errors['train'].append(mae_error_train)
mse_errors['train'].append(mse_error_train)
mape_errors['train'].append(mape_error_train)

true, predicted = model.evaluate('val')
mae_error_val, mse_error_val, mape_error_val = compute_errors(true, predicted)
mae_errors['val'].append(mae_error_val)
mse_errors['val'].append(mse_error_val)
mape_errors['val'].append(mape_error_val)

true, predicted = model.evaluate('test')
mae_error_test, mse_error_test, mape_error_test = compute_errors(true, predicted)
mae_errors['test'].append(mae_error_test)
mse_errors['test'].append(mse_error_test)
mape_errors['test'].append(mape_error_test)

print('MAE test:', mae_error_test)
print('MSE test:', mse_error_test)
print('MAPE test:', mape_error_test)
print('training time:', time.time() - start )

today = datetime.datetime.now().strftime('%Y-%m-%d')
model.save_pytorch_model('./{}-{}-{}-e128.pth'.format(dataname, today, str(model)) )
