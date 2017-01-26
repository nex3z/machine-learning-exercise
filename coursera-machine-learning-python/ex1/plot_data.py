import matplotlib.pyplot as plt


def plot_data(x, y, show=False):
    """
    Plots the data points x and y into a new figure.

    Parameters
    ----------
    x : array
        Data on x axis
    y : array
        Data on y axis
    show : bool
        True to show the plot immediately.
    """
    plt.figure()
    plt.plot(x, y, linestyle='', marker='x', color='r', label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    if show:
        plt.show()
