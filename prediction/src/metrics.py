import numpy as np


class Metrics:
    """
    Base class for evaluation metrics

    Parameters
    ----------
    x_true : numpy.array
        True labels of the predicted values
    x_pred : numpy.array
        The predicted values
    """

    def __init__(self, x_true=None, x_pred=None):
        self.x_true = x_true
        self.x_pred = x_pred

    def evaluate(self):
        """
        Returns the evaluated numerical value

        Returns
        -------
        float
            The evaluated value
        """
        raise NotImplementedError


class MAEMetrics(Metrics):
    """
    Mean absolute error
    """

    def __init__(self, x_true=None, x_pred=None):
        super().__init__(x_true, x_pred)

    def __str__(self):
        return "MAE"

    def evaluate(self):
        return np.average(np.abs(self.x_pred - self.x_true))


class MAPEMetrics(Metrics):
    """
    Mean absolute percentage error
    """

    def __init__(self, x_true=None, x_pred=None, mask_value=0):
        super().__init__(x_true, x_pred)

        self.mask_value = mask_value

    def __str__(self):
        return "MAPE"

    def MAPE_np(self):
        if self.mask_value is not None:
            mask = np.where(self.x_true > self.mask_value, True, False)  # only keep true value that is larger than mask value
            true = self.x_true[mask]
            pred = self.x_pred[mask]
        return np.mean(np.absolute(np.divide((true - pred), true)))

    def evaluate(self):
        return self.MAPE_np()


class MSEMetrics(Metrics):
    """
    Mean squared error
    """

    def __init__(self, x_true=None, x_pred=None):
        super().__init__(x_true, x_pred)

    def __str__(self):
        return "MSE"

    def evaluate(self):
        return np.average((self.x_true - self.x_pred) ** 2)


class RMSEMetrics(Metrics):
    """
    Root mean squared error
    """

    def __init__(self, x_true=None, x_pred=None):
        super().__init__(x_true, x_pred)

    def __str__(self):
        return "RMSE"

    def evaluate(self):
        return np.sqrt(np.average((self.x_true - self.x_pred) ** 2))


def compute_errors(true, predicted):
    metric = MAEMetrics(true, predicted)
    mae_error = metric.evaluate()

    metric = RMSEMetrics(true, predicted)
    rmse_error = metric.evaluate()

    metric = MAPEMetrics(true, predicted)
    mape_error = 100. * metric.evaluate()  # hundred percent
    return mae_error, rmse_error, mape_error


def compute_errors_all(model, mode=None):
    if mode is None:
        mode = ['train', 'val', 'test']
    errors = {mo: [] for mo in mode}
    for mo in mode:
        true, predicted = model.evaluate(mo)
        mae, rmse, mape = compute_errors(true, predicted)
        errors[mo].extend([mae, rmse, mape])

    return errors
