import matplotlib.pyplot as plt


def plot_data(x, y, show=False):
    """
    Plots the data points x and y into a new figure
    :param x: data on x axis
    :param y: data on y axis
    :param show: whether to show the plot
    """
    plt.figure()
    plt.plot(x, y, linestyle='', marker='x', color='r', label='Data Points')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    if show:
        plt.show()
