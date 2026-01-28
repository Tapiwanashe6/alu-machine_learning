#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.xlabel("Height (in)") # xlabel() - function from matplotlib.pyplot that is used to label an X-axis
plt.ylabel("Weight (lbs)") # ylabel() - function from matplotlib.pyplot that is used to label an Y-axis
plt.title("Men's Height vs Weight") # title() - function from matplotlib.pyplot that is used to label the title of the plot
plt.scatter(x, y, color="magenta", s=10)
plt.show()