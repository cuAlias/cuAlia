import math

import numpy
import numpy as np
import matplotlib.pyplot as plt

size = 8
x = np.arange(size)
a = np.random.random(size)
b = np.random.random(size)
c = np.random.random(size)

list = np.loadtxt(open("Pretreatment.csv","rb"),delimiter=",",skiprows=0)
list = list.transpose()
print(list)
a = list[0]
b = list[1]
c = list[2]
d = list[3]
a = np.log10(a)
b=  np.log10(b)
c = np.log10(c)
d = np.log10(d)

plt.figure(figsize=(10,5))#设置画布的尺寸
plt.title('Pretreatment time on 4090',fontsize=20)#标题，并设定字号大小
plt.xlabel(u'size(log scale)',fontsize=14)#设置x轴，并设定字号大小
plt.ylabel(u'ms(log scale)',fontsize=14)#设置y轴，并设定字号大小

plt.scatter(a,b,c='salmon')
plt.scatter(c,d,c='indianred')

plt.legend(['node2vec', 'sage/deepwalk'])#标签
plt.show()