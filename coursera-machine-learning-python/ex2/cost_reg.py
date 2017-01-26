from cost_function_reg import cost_function_reg


def cost_reg(theta, x, y, lam):
    j, g = cost_function_reg(theta.T, x, y, lam)
    return j.flatten()[0]

