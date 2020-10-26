import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))


def Cost(theta, x, y):
    part1 = np.log(Sigmoid(x.dot(theta))).dot(-y)
    part2 = np.log(1 - Sigmoid(x.dot(theta))).dot(1 - y)
    return (part1 - part2) / (len(x))


def Gradient(theta, x, y):
    return x.transpose().dot(Sigmoid(x.dot(theta)) - y) / len(x)


def Predict(parameter, x):
    return [1 if i >= 0.5 else 0 for i in Sigmoid(x.dot(parameter))]


def Handle():
    neg, pos, data = LoadData()
    data.insert(0, '辅助向量', 1)  # 线性代数技巧
    x = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1])
    theta = np.array(np.zeros(x.shape[1]))
    cost = Cost(theta, x, y)
    gradient = Gradient(theta, x, y)
    res = opt.minimize(fun=Cost, x0=theta, method='TNC', args=(x, y), jac=Gradient)  # fun和jac后面都是写函数名！！！
    learning_parameters = np.array([-25.1613186, 0.20623159, 0.20147149])
    predictions = Predict(learning_parameters, x)
    correct = []
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            correct.append(1)
        else:
            correct.append(0)
    accuracy = sum(correct) / len(x)
    return learning_parameters, accuracy


plt.figure(1)
plt.xlim(0,3)
plt.ylim(0,3)
plt.scatter([1,2],[1,2],marker='o')
plt.scatter([1,2],[2,1],marker='*')
plt.show()