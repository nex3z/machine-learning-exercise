import matplotlib.pyplot as plt


def draw_line(p1, p2, dash=False):
    """
    Draws a line from point p1 to point p2.

    Parameters
    ----------
    p1 : ndarray
        Point 1.
    p2 : ndarray
        Point 2.
    dash : bool
        True to plot dash line.
    """
    if dash:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--',color='k')
    else:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k')
