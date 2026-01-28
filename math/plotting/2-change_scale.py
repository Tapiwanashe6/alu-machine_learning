#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.title("Exponential Decay of C-14")
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.xlim(0, 28650) # Give limit to the axis - xlim()
plt.yscale("log") # Very important when you wanna scale the axis exponentiacally - yscale()
plt.plot(x, y)
plt.show()