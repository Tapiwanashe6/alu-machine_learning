# #!/usr/bin/env python3
# import numpy as np
# import matplotlib.pyplot as plt

# y = np.arange(0, 11) ** 3

# x = np.arange(0, 11)
# plt.plot(x, y, color="red")
# plt.show()
# import pyttsx3
# Text = input("Write any word: ")
# a = pyttsx3.init()
# b = a.say(Text)
# a.runAndWait()

def downward():
    count = 10
    for i in range(count):
        count -= 1
        print("*" * count)

downward()


