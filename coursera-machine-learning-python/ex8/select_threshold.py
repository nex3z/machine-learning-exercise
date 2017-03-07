import numpy as np


def select_threshold(y_val, p_val):
    """
    Find the best threshold (epsilon) to use for selecting outliers.

    Parameters
    ----------
    y_val : ndarray, shape (n_samples,)
        Labels from validation set, where n_samples is the number of samples and n_features is the number of features.
    p_val : ndarray, shape (n_samples,)
        The probability density from validation set.

    Returns
    -------
    best_epsilon : numpy.float64
        The best threshold for selecting outliers.
    best_F1 : numpy.float64
        The best F1 score.
    """
    step_size = (np.max(p_val) - np.min(p_val)) / 1000

    best_epsilon = 0.0
    best_F1 = 0.0

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = p_val < epsilon
        tp = np.sum(predictions[np.nonzero(y_val == True)])
        fp = np.sum(predictions[np.nonzero(y_val == False)])
        fn = np.sum(y_val[np.nonzero(predictions == False)] == True)
        if tp != 0:
            prec = 1.0 * tp / (tp + fp)
            rec = 1.0 * tp / (tp + fn)
            F1 = 2.0 * prec * rec / (prec + rec)
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = epsilon

    return best_epsilon, best_F1
