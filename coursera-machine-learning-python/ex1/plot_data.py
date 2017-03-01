import matplotlib.pyplot as plt


def plot_data(x, y):
    """
    Plots the data points x and y.

    Parameters
    ----------
    x : array-like
        Data on x axis.
    y : array-like
        Data on y axis.
    """
    plt.plot(x, y, linestyle='', marker='x', color='r', label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
