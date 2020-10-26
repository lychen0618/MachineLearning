import numpy as np
import matplotlib.pyplot as plt

from regression import preprocess
import csv

preprocess.pearson([1, 0, 1.1], [1, 1.9, 3])
preprocess.spearman([1, 2, 3], [1, 1.9, 3])

dataArray=[]
with open(r'..\data\regression\boston_house_prices.csv', encoding="utf-8") as f:
    reader = csv.reader(f)
    for i in reader:
        temp=i[0].strip().split()
        dataArray.append(temp)
    #print(dataArray)

dataArray = [[float(x) for x in row] for row in dataArray]  # 将数据从string形式转换为float形式

dataArray = np.array(dataArray)  # 将list数组转化成array数组便于查看数据结构
print(dataArray.shape)  # 利用.shape查看结构

plt.subplot(1,3,1)
plt.scatter(dataArray[:,5],dataArray[:,13],s = 20)
plt.ylabel("MEDV")
plt.xlabel("RM")


plt.subplot(1,3,2)
plt.scatter(dataArray[:,10],dataArray[:,13],s = 20)
plt.ylabel("MEDV")
plt.xlabel("PTRATIO")


plt.subplot(1,3,3)
plt.scatter(dataArray[:,12],dataArray[:,13],s = 20)
plt.ylabel("MEDV")
plt.xlabel("LSTAT")
plt.show()
