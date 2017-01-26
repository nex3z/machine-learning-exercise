from cost_function_reg import cost_function_reg


def gradient_reg(theta, x, y, lam):
    j, g = cost_function_reg(theta.T, x, y, lam)
    # print "gradient(): ", g.flatten().shape
    return g.flatten()
