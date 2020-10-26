import numpy as np
from decimal import Decimal


def gradient_descent(x_train, y_train, alpha, type, lam=0):
    m = x_train.shape[0]
    n = x_train.shape[1]
    if type < 3:
        theta = np.zeros((n + 1, 1))
    else:
        theta = np.ones((n + 1, 1))

    # print(theta.shape)
    addcol = np.ones((m, 1))
    x_train = np.c_[addcol, x_train]

    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train[:, 0].shape)

    # method1
    # times = 0
    # while times < 100:
    #     times += 1
    #     temp = theta
    #     for j in range(n + 1):
    #         arr=[]
    #         for i in range(m):
    #             nsum=0.0
    #             for ii in range(n+1):
    #                 nsum+=x_train[i][ii]*theta[ii][0]
    #             arr.append(y_train[i]-nsum)
    #         nsum=0.0
    #         for i in range(m):
    #             nsum+=x_train[i][j]*arr[i]
    #         temp[j][0] = temp[j][0] + alpha * nsum
    #     #print(temp)
    #     theta = temp

    # method2
    times = 0
    while times < 100:
        times += 1
        temp = np.zeros((n + 1, 1))
        for j in range(n + 1):
            temp[j][0] = theta[j][0]
        for j in range(n + 1):
            if type == 1:
                theta[j][0] = theta[j][0] + alpha * np.dot(y_train - np.dot(x_train, theta).reshape(m, ), x_train[:, j])
            elif type == 2:
                theta[j][0] = theta[j][0] + alpha * (np.dot(y_train - np.dot(x_train, theta).reshape(m, ),
                                                            x_train[:, j]) - 2 * lam * theta[j][0])
            else:
                another = 0.0
                if Decimal(str(theta[j][0])) > Decimal('0.0'):
                    another = lam
                elif Decimal(str(theta[j][0])) < Decimal('0.0'):
                    another = -lam
                theta[j][0] = theta[j][0] + alpha * (np.dot(y_train - np.dot(x_train, theta).reshape(m, ),
                                                            x_train[:, j]) - another)

        diff = Decimal('0.0')
        for j in range(n + 1):
            a = Decimal(str(theta[j][0]))
            b = Decimal(str(temp[j][0]))
            diff += Decimal(a - b) if a > b else Decimal(b - a)
            # print(diff)
        if diff < Decimal('0.001'):
            break
    return theta


class LinearRegression:
    theta = []

    def fit(self, x_train, y_train):
        self.theta = gradient_descent(x_train, y_train, 0.001, 1)
        print(self.theta)

    def predict(self, x_test):
        return (np.dot(x_test, self.theta[1:]) + self.theta[0][0]).reshape(x_test.shape[0], )


class RidgeRegression:
    theta = []
    alpha = 0

    def __init__(self, alpha=0):
        self.alpha = alpha

    def fit(self, x_train, y_train):
        self.theta = gradient_descent(x_train, y_train, 0.001, 2, self.alpha)
        # print(self.theta)

    def predict(self, x_test):
        return (np.dot(x_test, self.theta[1:]) + self.theta[0][0]).reshape(x_test.shape[0], )


class LassoRegression:
    theta = []
    alpha = 0

    def __init__(self, alpha=0):
        self.alpha = alpha

    def fit(self, x_train, y_train):
        self.theta = gradient_descent(x_train, y_train, 0.001, 3, self.alpha)
        print(self.theta)

    def predict(self, x_test):
        return (np.dot(x_test, self.theta[1:]) + self.theta[0][0]).reshape(x_test.shape[0], )


class DecisiontreeRegression:
    x = 0
