from cost_function import cost_function

def cost(theta, x, y):
    j, g = cost_function(theta.T, x, y)
    return j.flatten()[0]

