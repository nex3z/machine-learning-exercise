from cost_function import cost_function


def gradient(theta, x, y):
    j, g = cost_function(theta.T, x, y)
    # print "gradient(): ", g.flatten().shape
    return g.flatten()
