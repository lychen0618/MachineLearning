from sklearn.model_selection import train_test_split
from regression.preprocess import *
import matplotlib.pyplot as plt
import numpy as np

# 读取数据，选择特征
x_data, y_data = preprocess(r'..\data\regression\boston_house_prices.csv')
print(x_data.shape)
print(y_data.shape)
# print(x_data)
# print(y_data)

# 随机擦痒20%的数据构建测试样本，剩余作为训练样本
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=0, test_size=0.20)
print(x_train.shape)
print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# 0均值标准化
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test /= std

# mean0 = y_train.mean(axis=0)
# y_train -= mean0
# std0 = y_train.std(axis=0)
# y_train /= std0
#
# y_test -= mean0
# y_test /= std0

# mean1=np.tile(np.transpose(mean),(x_train.shape[0],1))
# x_train -= mean1
# std1=np.tile(np.transpose(std),(x_train.shape[0],1))
# x_train /= std1
# mean2=np.tile(np.transpose(mean),(x_test.shape[0],1))
# std2=np.tile(np.transpose(std),(x_test.shape[0],1))
# x_test -= mean2
# x_test /= std2

# 使用线性回归模型LinearRegression对波士顿房价数据进行训练及预测
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# 使用训练数据进行参数估计
lr.fit(x_train, y_train)
# 回归预测
lr_y_predict = lr.predict(x_test)
print(lr_y_predict - y_test)

# 使用线性回归模型Ridge对波士顿房价数据进行训练及预测
from sklearn.linear_model import Ridge

rr = Ridge(alpha=0.5)
rr.fit(x_train, y_train)
rr_y_predict = rr.predict(x_test)
print(rr_y_predict - y_test)

# 使用线性回归模型Lasso对波士顿房价数据进行训练及预测
from sklearn.linear_model import Lasso

lar = Lasso(alpha=0.5)
lar.fit(x_train, y_train)
lar_y_predict = lar.predict(x_test)
print(lar_y_predict - y_test)

# 使用决策树回归模型对波士顿房价数据进行训练及预测
from sklearn.tree import DecisionTreeRegressor

der = DecisionTreeRegressor()
der.fit(x_train, y_train)
der_y_predict = der.predict(x_test)
print(der_y_predict - y_test)

import regression.regression_method as myre

mylr = myre.LinearRegression()
mylr.fit(x_train, y_train)
mylr_y_predict = mylr.predict(x_test)
print(mylr_y_predict - y_test)

myrr = myre.RidgeRegression(0.5)
myrr.fit(x_train, y_train)
myrr_y_predict = myrr.predict(x_test)
print(myrr_y_predict - y_test)

mylar = myre.RidgeRegression(0.5)
mylar.fit(x_train, y_train)
mylar_y_predict = mylar.predict(x_test)
print(mylar_y_predict - y_test)

# plt.subplot(1,4,1)
# plt.scatter(y_test,lr_y_predict-y_test,s = 10)
# plt.ylabel("lr_y_predict - y_test")
# plt.xlabel("y_test")
#
# plt.subplot(1,4,2)
# plt.scatter(y_test,rr_y_predict-y_test,s = 10)
# plt.ylabel("rr_y_predict - y_test")
# plt.xlabel("y_test")
#
# plt.subplot(1,4,3)
# plt.scatter(y_test,lar_y_predict-y_test,s = 10)
# plt.ylabel("lar_y_predict - y_test")
# plt.xlabel("y_test")
#
# plt.subplot(1,4,4)
# plt.scatter(y_test,der_y_predict-y_test,s = 10)
# plt.ylabel("der_y_predict - y_test")
# plt.xlabel("y_test")
# plt.show()

plt.subplot(1,4,1)
plt.scatter(y_test,mylr_y_predict-y_test,s = 10)
plt.ylabel("mylr_y_predict - y_test")
plt.xlabel("y_test")

plt.subplot(1,4,2)
plt.scatter(y_test,myrr_y_predict-y_test,s = 10)
plt.ylabel("myrr_y_predict - y_test")
plt.xlabel("y_test")

plt.subplot(1,4,3)
plt.scatter(y_test,mylar_y_predict-y_test,s = 10)
plt.ylabel("mylar_y_predict - y_test")
plt.xlabel("y_test")
plt.show()



from sklearn.metrics import r2_score

print("sk_lr:%f" % r2_score(y_test, lr_y_predict))
print("sk_rr:%f" % r2_score(y_test, rr_y_predict))
print("sk_lar:%f" % r2_score(y_test, lar_y_predict))
print("sk_der:%f" % r2_score(y_test, der_y_predict))
print("my_lr:%f" % r2_score(y_test, mylr_y_predict))
print("my_rr:%f" % r2_score(y_test, myrr_y_predict))
print("my_lar:%f" % r2_score(y_test, mylar_y_predict))
