import math

import numpy
import numpy as np
import matplotlib.pyplot as plt

size = 8
x = np.arange(size)
a = np.random.random(size)
b = np.random.random(size)
c = np.random.random(size)

list = np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=0)
list = list.transpose()
print(list)
a = list[0]
b = list[1]
c = list[2]
# a = numpy.log2(a)
# b = numpy.log2(b)
# c = numpy.log2(c)


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2


plt.title("Deepwalk sampling")
plt.bar(x, a,  width=width, label='our work', color= (62/255, 134/255, 181/255))
plt.bar(x + width, b, width=width, label='nextdoor', color= (149/255, 167/255, 126/255))
plt.bar(x + 2 * width, c, width=width, label='skywalker', color= (229/255, 190/255, 121/255))
plt.legend()
plt.show()
