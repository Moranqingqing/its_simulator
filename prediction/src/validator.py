import datetime
import os
from shutil import rmtree

from scipy import stats

from src.metrics import *
from src.prediction_models.neural_nets.nn_predictor import NNPredictor


class Validator:
    """
    Evaluator that evaluates the performance of the sow5 models

    Parameters
    ----------
    predictor : PredictionModel
        The sow4 model to be evaluated
    traffic_data
        The traffic data used in training/validation/testing
    metrics : [Metrics]
        The metrics to use during evaluation,
        default: all metrics available
    param_dict : dict
        hyperparameters to search
    test_data
        Unseen test data (optional)

    Attributes
    ----------
    predictor : PredictionModel
        The sow4 model to be evaluated
    metrics : [Metrics]
        The metrics to use during evaluation,
        default: all metrics available
    """

    def __init__(self, predictor, traffic_data, param_dict, metrics=None, test_data=None):
        self.predictor = predictor
        self.traffic_data = traffic_data
        self.param_dict = param_dict
        self.test_data = test_data
        if not metrics:
            self.metrics = [MAEMetrics, MAPEMetrics, MSEMetrics, RMSEMetrics]
        else:
            self.metrics = metrics

    def cross_validate(self, k, val_pct):
        """
        Evaluates the predictor using K fold cross validation

        Parameters
        ----------
        k : int
            number of splits
        val_pct : float
            percentage of data to hold as validation set

        Returns
        -------
        dict
            Results for each predictor mapped to all the metrics
        """
        results = {}
        predictor_str = self.predictor.__name__
        results[predictor_str] = {}

        for metrics in self.metrics:
            results[predictor_str][metrics().__str__()] = []

        length = len(self.traffic_data.timestamps)

        # can use the whole dataset for training/validation since a completely new test set is provided
        if self.test_data:
            test_length = 0
        else:
            test_length = int(length / k)

        val_length = int((length - test_length) * val_pct)

        for i in range(k):
            test_timestamps = self.traffic_data.timestamps[test_length * i:test_length * (i + 1)]
            non_test_timestamps = [j for j in self.traffic_data.timestamps if j not in test_timestamps]
            train_timestamps = non_test_timestamps[:-val_length]
            val_timestamps = non_test_timestamps[-val_length:]
            self.traffic_data.split_by_timestamp(train_timestamps, val_timestamps, test_timestamps)
            if self.test_data:
                self.traffic_data.test_data = self.test_data.data
            params = self.tune_model(trace=False)
            model = self.predictor(traffic_data=self.traffic_data, **params)
            if isinstance(model, NNPredictor):
                model.save_model = True
                cur_time = str(datetime.datetime.now())
                os.mkdir('./' + cur_time + '/')
                model.save_path = './' + cur_time + '/'
                model.max_epoch = 2000
                model.save('./' + cur_time + '/')

                model.train()

                min_epoch = None
                min_loss = 2e32
                path = next(os.walk('./' + cur_time + '/'))[1][0]
                for epoch in range(50, 2050, 50):
                    model.load_pytorch_model('./' + cur_time + '/' + path + '/e{}.pt'.format(epoch))
                    x_true, x_pred = model.evaluate('val')
                    if MSEMetrics(x_true, x_pred).evaluate() < min_loss:
                        min_epoch = epoch
                model.load_pytorch_model('./' + cur_time + '/' + path + '/e{}.pt'.format(min_epoch))
                rmtree('./' + cur_time + '/', True)

            else:
                model.train()
            x_true, x_pred = model.evaluate('test')
            eval_metrics = [metrics(x_true, x_pred) for metrics in self.metrics]
            for metrics in eval_metrics:
                results[predictor_str][metrics.__str__()].append(metrics.evaluate())
        for metrics in self.metrics:
            samples = results[predictor_str][metrics().__str__()]
            mean = np.mean(samples)
            ci_low, ci_high = stats.t.interval(0.95, len(samples), loc=mean, scale=stats.sem(samples))
            results[predictor_str][metrics().__str__()] = {'mean': mean, 'CI': [ci_low, ci_high]}
        return results

    def tune_model(self, trace=True, return_best_model=False):
        best_model = None
        params = {param: vals[0] for param, vals in self.param_dict.items()}
        for param in self.param_dict:
            if len(self.param_dict[param]) > 1:
                min_error = float("inf")
                min_val = None
                for val in self.param_dict[param]:
                    params[param] = val
                    model = self.predictor(traffic_data=self.traffic_data, **params)
                    print(str(model))
                    try:
                        model.train()
                    except Exception as e:
                        print(e)
                        continue
                    x_true, x_pred = model.evaluate('val')
                    error = RMSEMetrics(x_true, x_pred).evaluate()
                    if trace:
                        print(params, error)
                    if error < min_error:
                        min_val = val # min_val == best hyperparameter
                        min_error = error
                        best_model = model
                if min_val is None:
                    raise RuntimeError
                params[param] = min_val # best hyperparameter

        if return_best_model:
            return params, best_model
        return params
