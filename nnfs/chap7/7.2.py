#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 2*x**2

# np.arange(start, stop, step) to get a smoother line
x = np.arange(0, 5, 0.001)
y = f(x)
plt.plot(x, y)


# the point and the "close enough" point
p2_delta = 0.0001
x1 = 2
x2 = x1+p2_delta

y1 = f(x1)
y2 = f(x2)

print((x1, y1), (x2, y2))

# derivate approximation and y-intercept for the tangent line
approximate_derivative = (y2-y1)/(x2-x1)
b = y2 - approximate_derivative*x2

# We put the tangent line calculation info a function so we can call it
# multiple times for different values of x
# approximate_derivative and b are constant for given function
# thus calculated once above this function

def tangent_line(x):
    return approximate_derivative*x + b

# Plotting the tangent line
# +/- 0.9 to draw the tangent line on our graph
# then we calculate the y for given x using the tangent line function
# matplotlib will draw a line for us through these points
to_plot = [x1-0.9, x1, x1+0.9]
plt.plot(to_plot, [tangent_line(i) for i in to_plot])

print('Approximate derivative for f(x)', f'where x = {x1} is {approximate_derivative}')

plt.show()
