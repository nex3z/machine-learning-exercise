import numpy as np


from train_linear_reg import train_linear_reg


def validation_curve(x, y, x_val, y_val):
    """
    Generate the train and validation errors needed to plot a validation curve that we can use to select lambda.

    Parameters
    ----------
    X : ndarray, shape (n_train_samples, n_features)
        Samples of training set, where n_train_samples is the number of samples in training set and n_features is the
        number of features.
    y : ndarray, shape (n_train_samples,)
        Labels of training set.
    X_val : ndarray, shape (n_val_samples, n_features)
        Samples of cross validation set, where n_val_samples is the number of samples in cross validation set.
    y_val : ndarray, shape (n_val_samples,)
        Labels of cross validation set.

    Returns
    -------
    lambda_vec : ndarray, shape (n_lambdas,)
        The regularization parameters used for models, where n_lambdas is the number of lambda values used for the
        validation curve.
    error_train : ndarray, shape (n_lambdas,)
        Train set error.
    error_val : ndarray, shape (n_lambdas,)
        Cross validation set error.
    """
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))
    m = x.shape[0]
    m_val = x_val.shape[0]
    for i in range(len(lambda_vec)):
        l = lambda_vec[i]
        theta = train_linear_reg(x, y, l)
        error_train[i] = 1.0 / (2 * m) * np.sum(np.square(x.dot(theta) - y))
        error_val[i] = 1.0 / (2 * m_val) * np.sum(np.square(x_val.dot(theta) - y_val))

    return lambda_vec, error_train, error_val
