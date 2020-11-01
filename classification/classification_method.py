from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def gradient_descent(x_train, y_train, alpha, type):
    m = x_train.shape[0]
    n = x_train.shape[1]
    theta = np.zeros((n + 1, 1))

    addcol = np.ones((m, 1))
    x_train = np.c_[addcol, x_train]

    if type == 1:
        times = 0
        while True:
            anotherTemp = np.dot(x_train, theta).reshape(m, )
            # print(anotherTemp)
            h = np.zeros(m)

            for i in range(m):
                s = str(anotherTemp[i])
                if not (Decimal('0.0') > Decimal(s)):
                    h[i] = 1
            if np.all(h == y_train) or times > 1000:
                break

            for i in range(m):
                if y_train[i] != h[i]:
                    for k in range(n + 1):
                        theta[k][0] += alpha * (y_train[i] - h[i]) * x_train[i, k]
                    break
            times += 1
    else:
        for times in range(2000):
            out = Sigmoid(x_train.dot(theta))
            # print(out.shape)
            error = np.reshape(y_train, out.shape) - out
            # print(theta)
            theta += alpha * (x_train.T.dot(error))
            # print(theta)

    return np.reshape(theta, (n + 1))


def Indicate(z):
    return np.array([1 if Decimal(str(i)) >= Decimal('0.0') else 0 for i in z])


def Cost0(theta, x, y):
    part1 = np.log(Indicate(x.dot(theta))).dot(-y)
    part2 = np.log(1 - Indicate(x.dot(theta))).dot(1 - y)
    return part1 - part2


def Gradient0(theta, x, y):
    return x.transpose().dot(Indicate(x.dot(theta)) - y)


def perceptron(x1, x2):
    x = np.vstack((x1, x2))
    y = np.append(np.ones((x1.shape[0])), np.zeros((x2.shape[0])))
    m = x.shape[0]
    n = x.shape[1]
    theta = np.zeros((n + 1, 1))

    addcol = np.ones((m, 1))
    x = np.c_[addcol, x]
    res = opt.minimize(fun=Cost0, x0=theta, method='TNC', args=(x, y), jac=Gradient0)
    return res.x


# def perceptron(x1, x2):
#     x = np.vstack((x1, x2))
#     y = np.append(np.ones((x1.shape[0])), np.zeros((x2.shape[0])))
#     return gradient_descent(x, y, 0.1, 1)


def Sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def Cost(theta, x, y):
    part1 = np.log(Sigmoid(x.dot(theta))).dot(-y)
    part2 = np.log(1 - Sigmoid(x.dot(theta))).dot(1 - y)
    return part1 - part2


def Gradient(theta, x, y):
    return x.transpose().dot(Sigmoid(x.dot(theta)) - y)


def logistic_regression(x1, x2):
    x = np.vstack((x1, x2))
    y = np.append(np.ones((x1.shape[0])), np.zeros((x2.shape[0])))
    m = x.shape[0]
    n = x.shape[1]
    theta = np.zeros((n + 1, 1))

    addcol = np.ones((m, 1))
    x = np.c_[addcol, x]
    res = opt.minimize(fun=Cost, x0=theta, method='TNC', args=(x, y), jac=Gradient)
    return res.x


# def logistic_regression(x1, x2):
#     x = np.vstack((x1, x2))
#     y = np.append(np.ones((x1.shape[0])), np.zeros((x2.shape[0])))
#     return gradient_descent(x, y, 0.01, 2)


def myNewLDA(x, y, k):
    # x为数据集，y为label，k为目标维数
    label_ = list(set(y))
    x_classify = {}

    for label in label_:
        xi = np.array([x[i] for i in range(len(x)) if y[i] == label])
        x_classify[label] = xi

    miu = np.mean(x, axis=0)
    miu_classify = {}

    for label in label_:
        miui = np.mean(x_classify[label], axis=0)
        miu_classify[label] = miui

    # 计算类内散度矩阵
    Sw = np.zeros((len(miu), len(miu)))
    for label in label_:
        Sw += np.dot((x_classify[label] - miu_classify[label]).T,
                     x_classify[label] - miu_classify[label])

    # 计算类间散度矩阵
    Sb = np.zeros((len(miu), len(miu)))
    for label in label_:
        Sb += len(x_classify[label]) * np.dot((miu_classify[label] - miu).reshape(
            (len(miu), 1)), (miu_classify[label] - miu).reshape((1, len(miu))))

    # 计算Sw-1*Sb的特征值和特征矩阵
    # U, S, V = np.linalg.svd(Sw)
    # S = np.diag(S)
    # SW_inverse = V.dot(np.linalg.pinv(S)).dot(U.T)
    # A = SW_inverse.dot(Sb)
    # eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    # 按从小到大排序，输出排序指示值
    sorted_indices = np.argsort(eig_vals)
    # 反转
    sorted_indices = sorted_indices[::-1]
    # 提取前k个特征向量
    topk_eig_vecs = eig_vecs[:, sorted_indices[0:k:1]]
    # s[i:j:k]，i起始位置,j终止位置，k表示步长，默认为1
    # s[::-1]是从最后一个元素到第一个元素复制一遍（反向）

    return topk_eig_vecs


def gaussian_kernel_func(x, z):
    # sigma = 2
    # return math.exp(-(x - z).dot((x - z).T) / (2 * sigma * sigma))
    return x.dot(z) ** 2


def LDAWithKernel(x, y, k):
    # x为数据集，y为label，k为目标维数
    label_ = list(set(y))
    x_classify = {}
    for label in label_:
        xi = np.array([x[i] for i in range(len(x)) if y[i] == label])
        x_classify[label] = xi

    M = np.array([[gaussian_kernel_func(x[i], x[j]) for j in range(x.shape[0])] for i in range(x.shape[0])])

    m = np.mean(M, axis=1)
    m_classify = {}

    for label in label_:
        mi = np.mean([M[i] for i in range(x.shape[0]) if y[i] == label], axis=0)
        m_classify[label] = mi

    # 计算类内散度矩阵
    Sw = M.dot(M.T)
    for label in label_:
        Sw -= len(x_classify[label]) * m_classify[label].T.dot(m_classify[label])

    # 计算类间散度矩阵
    Sb = np.zeros((len(m), len(m)))
    for label in label_:
        Sb += len(x_classify[label]) * np.dot((m_classify[label] - m).reshape(
            (len(m), 1)), (m_classify[label] - m).reshape((1, len(m))))

    # 计算Sw-1*Sb的特征值和特征矩阵
    # U, S, V = np.linalg.svd(Sw)
    # S = np.diag(S)
    # SW_inverse = V.dot(np.linalg.pinv(S)).dot(U.T)
    # A = SW_inverse.dot(Sb)
    # eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    # 按从小到大排序，输出排序指示值
    sorted_indices = np.argsort(eig_vals)
    # 反转
    sorted_indices = sorted_indices[::-1]
    # 提取前k个特征向量
    topk_eig_vecs = eig_vecs[:, sorted_indices[0:k:1]]
    # s[i:j:k]，i起始位置,j终止位置，k表示步长，默认为1
    # s[::-1]是从最后一个元素到第一个元素复制一遍（反向）

    temp = {}
    for label in label_:
        temp[label] = np.dot([M[i] for i in range(x.shape[0]) if y[i] == label], topk_eig_vecs)
        temp[label] = np.real(np.mean(temp[label], axis=0))
        for j in range(len(temp[label])):
            temp[label][j] = 1 if temp[label][j] > 0 else 0

    return topk_eig_vecs, temp


def myLDA(x, y, k):
    # x为数据集，y为label，k为目标维数
    label_ = list(set(y))
    x_classify = {}

    for label in label_:
        xi = np.array([x[i] for i in range(len(x)) if y[i] == label])
        x_classify[label] = xi

    miu = np.mean(x, axis=0)
    miu_classify = {}

    for label in label_:
        miui = np.mean(x_classify[label], axis=0)
        miu_classify[label] = miui

    # 计算类内散度矩阵
    Sw = np.zeros((len(miu), len(miu)))
    for label in label_:
        Sw += np.dot((x_classify[label] - miu_classify[label]).T,
                     x_classify[label] - miu_classify[label])

    # 计算类间散度矩阵
    Sb = np.zeros((len(miu), len(miu)))
    for label in label_:
        Sb += len(x_classify[label]) * np.dot((miu_classify[label] - miu).reshape(
            (len(miu), 1)), (miu_classify[label] - miu).reshape((1, len(miu))))

    # 计算Sw-1*Sb的特征值和特征矩阵
    # U, S, V = np.linalg.svd(Sw)
    # S = np.diag(S)
    # SW_inverse = V.dot(np.linalg.pinv(S)).dot(U.T)
    # A = SW_inverse.dot(Sb)
    # eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)

    # 按从小到大排序，输出排序指示值
    sorted_indices = np.argsort(eig_vals)
    # 反转
    sorted_indices = sorted_indices[::-1]
    # 提取前k个特征向量
    topk_eig_vecs = eig_vecs[:, sorted_indices[0:k:1]]
    # s[i:j:k]，i起始位置,j终止位置，k表示步长，默认为1
    # s[::-1]是从最后一个元素到第一个元素复制一遍（反向）

    temp = {}
    for i in label_:
        temp[i] = np.dot(x_classify[i], topk_eig_vecs)
        temp[i] = np.real(np.mean(temp[i], axis=0))
        for j in range(len(temp[i])):
            temp[i][j] = 1 if temp[i][j] > 0 else 0
    print(temp)
    return topk_eig_vecs, temp


if '__main__' == __name__:
    iris = load_iris()
    X = iris.data
    # print(X.shape)
    # print(X)
    y = iris.target
    # print(y)
    W = myLDA(X, y, 2)
    X_new = np.dot((X), W)  # 估计值

    plt.figure(1)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.title("LDA reducing dimension - our method")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # 与sklearn中的LDA函数对比
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    X_new = - X_new  # 为了对比方便，取个相反数，并不影响分类结果
    # print(X_new)
    plt.figure(2)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    plt.title("LDA reducing dimension - sklearn method")
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.show()
