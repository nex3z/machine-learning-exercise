import numpy as np


def select_threshold(yval, pval):
    """
    Find the best threshold (epsilon) to use for selecting outliers.

    Parameters
    ----------
    yval : ndarray, shape (n_samples, n_features)
        Labels from validation set, where n_samples is the number of samples and n_features is the number of features.
    pval : ndarray, shape (n_samples, n_features)
        The probability density from validation set.

    Returns
    -------
    best_epsilon : numpy.float64
        The best threshold for selecting outliers.
    best_F1 : numpy.float64
        The best F1 score.
    """
    step_size = (np.max(pval) - np.min(pval)) / 1000

    best_epsilon = 0.0
    best_F1 = 0.0

    for epsilon in np.arange(min(pval), max(pval), step_size):
        predictions = pval < epsilon
        tp = np.sum(predictions[np.nonzero(yval.ravel() == True)])
        fp = np.sum(predictions[np.nonzero(yval.ravel() == False)])
        fn = np.sum(yval[np.nonzero(predictions.ravel() == False)] == True)
        if tp != 0:
            prec = 1.0 * tp / (tp + fp)
            rec = 1.0 * tp / (tp + fn)
            F1 = 2.0 * prec * rec / (prec + rec)
            if F1 > best_F1:
                best_F1 = F1
                best_epsilon = epsilon

    return best_epsilon, best_F1
