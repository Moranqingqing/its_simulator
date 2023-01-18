from scipy import stats

from src.metrics import *
from src.prediction_models.neural_nets.nn_predictor import NNPredictor


class Evaluator:
    """
    Evaluator that evaluates the performance of the sow5 models

    Parameters
    ----------
    predictors : [PredictionModel]
        The sow5 models to be evaluated
    metrics : [Metrics]
        The metrics to use during evaluation,
        default: all metrics available

    Attributes
    ----------
    predictors : [PredictionModel]
        The sow5 models to be evaluated
    metrics : [Metrics]
        The metrics to use during evaluation,
        default: all metrics available
    """

    def __init__(self, predictors, metrics=None):
        self.predictors = predictors
        if not metrics:
            self.metrics = [MAEMetrics, MAPEMetrics, MSEMetrics, RMSEMetrics]
        else:
            self.metrics = metrics

    def train(self):
        """
        Trains all the predictors

        Returns
        -------
        None
        """
        for predictor in self.predictors:
            predictor.train()

    def evaluate(self, links=None):
        """
        Evaluates the predictors on the test set

        Parameters
        ----------
        links : list of str
            which links to include in the calculation of metrics

        Returns
        -------
        dict
            Results for each predictor mapped to all the metrics
        """
        results = {}
        for predictor in self.predictors:
            if isinstance(predictor, NNPredictor):
                predictor_str = predictor.__str__() + '-e{}'.format(predictor.max_epoch)
            else:
                predictor_str = predictor.__str__()
            results[predictor_str] = {}
            x_true, x_pred = predictor.evaluate('test', links)
            eval_metrics = [metrics(x_true, x_pred) for metrics in self.metrics]
            for metrics in eval_metrics:
                results[predictor_str][metrics.__str__()] = metrics.evaluate()
        return results
