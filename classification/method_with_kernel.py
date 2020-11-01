import math
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np


class LDAWithKernel:
    x_train = []
    alpha = []
    temp = {}

    def kernel_func(self, x, z):
        return x.dot(z) ** 2

    def fit(self, x, y, k):
        # x为数据集，y为label，k为目标维数
        self.x_train = x.T

        label_ = list(set(y))

        M = np.array(self.kernel_func(x, self.x_train))

        m_classify = {}
        for label in label_:
            mi = np.array([M[i] for i in range(len(x)) if y[i] == label])
            m_classify[label] = mi

        m = np.mean(M, axis=1)
        m_mean = {}

        for label in label_:
            mi = np.mean(m_classify[label], axis=0)
            m_mean[label] = mi

        # 计算类内散度矩阵
        Sw = M.dot(M.T)
        for label in label_:
            Sw -= m_classify[label].shape[0] * np.dot(m_mean[label].reshape((len(m), 1)),
                                                      m_mean[label].reshape((1, len(m))))

        Sw += 0.1 * np.identity(len(m))

        # 计算类间散度矩阵
        Sb = np.zeros((len(m), len(m)))
        for label in label_:
            Sb += len(m_classify[label]) * np.dot((m_mean[label] - m).reshape(
                (len(m), 1)), (m_mean[label] - m).reshape((1, len(m))))

        # 计算Sw-1*Sb的特征值和特征矩阵
        # U, S, V = np.linalg.svd(Sw)
        # S = np.diag(S)
        # SW_inverse = V.dot(np.linalg.pinv(S)).dot(U.T)
        # A = SW_inverse.dot(Sb)
        # eig_vals, eig_vecs = np.linalg.eig(A)
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)

        # # 按从小到大排序，输出排序指示值
        # sorted_indices = np.argsort(eig_vals)
        # # 反转
        # sorted_indices = sorted_indices[::-1]
        # 提取前k个特征向量
        self.alpha = eig_vecs[:, 0:k:1]
        # s[i:j:k]，i起始位置,j终止位置，k表示步长，默认为1
        # s[::-1]是从最后一个元素到第一个元素复制一遍（反向）

        for label in label_:
            self.temp[label] = np.dot(m_classify[label], self.alpha)
            self.temp[label] = np.mean(self.temp[label], axis=0)
            for j in range(self.temp[label].shape[0]):
                self.temp[label][j] = 1 if self.temp[label][j] > 0 else 0

    def predict(self, x_test):
        midvalue = np.array(self.kernel_func(x_test, self.x_train)).dot(self.alpha)
        mylda_y_predict = np.zeros(x_test.shape[0])
        for i in range(midvalue.shape[0]):
            for j in range(len(midvalue[i])):
                midvalue[i][j] = 1 if midvalue[i][j] > 0 else 0

        for i in range(midvalue.shape[0]):
            equalCount = 0
            for k in self.temp.keys():
                tec = 0
                for j in range(len(self.temp[k])):
                    tec += 1 if midvalue[i][j] == self.temp[k][j] else 0
                if tec > equalCount:
                    equalCount = tec
                    mylda_y_predict[i] = k

        return mylda_y_predict
