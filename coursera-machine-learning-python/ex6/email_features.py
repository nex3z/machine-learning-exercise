import numpy as np


def email_features(word_indices):
    """
    Takes in a word_indices vector and produces a feature vector from the word indices.

    Parameters
    ----------
    word_indices : array-like
        List of word indices.

    Returns
    -------
    ndarray
        Feature vector from word indices.
    """
    # Total number of words in the dictionary
    n = 1899

    x = np.zeros((n, 1))
    x[word_indices] = 1

    return x
