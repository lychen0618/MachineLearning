import math
from decimal import Decimal

import numpy as np
from classification.method_with_kernel import *
import regression.regression_method as myre
import classification.classification_method as mycf
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm


def gaussian_kernel_func(x, z):
    return x.dot(z) ** 2


# 从文件读取数据
def extractData(fileName, size, type):
    file = open(fileName, "rb")
    if type == 1:
        file.seek(16)
        dataMat = np.zeros((size, 784), dtype=np.float64)
        for i in range(size):
            for j in range(28 * 28):
                pixel = file.read(1)
                dataMat[i][j] = ord(pixel) * 1.0 / 255
    else:
        file.seek(8)
        dataMat = np.zeros(size)
        for i in range(size):
            dataMat[i] = ord(file.read(1))
    file.close()
    return dataMat


# cut变换提取特征
def cutTransform(dataMat, div):
    size = dataMat.shape[0]
    dataMatAdvance = np.zeros((size, div * div), dtype=np.float64)
    l = 28 / div
    for i in range(size):
        for r in range(28):
            for c in range(28):
                dataMatAdvance[i][int(r / l) * div + int(c / l)] += dataMat[i][r * 28 + c]

    dataMatAdvance /= l * l
    return dataMatAdvance


def accuracyCount(predict, real):
    count = 0
    size = predict.shape[0]
    for i in range(size):
        if (round(predict[i]) == real[i]):
            count += 1
    return count * 1.0 / size


def Sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def handwritingClassfy(percent):
    # percent代表选择的原始数据集百分之几
    # trainSize：训练集的大小
    # testSize：测试集的大小
    trainSize = int(60000 * percent)
    testSize = int(10000 * percent)
    # trainMat0是训练集数字图像数据数组，每一行784列，代表一幅数字图像
    trainMat0 = extractData(r"..\data\knn\train-images.idx3-ubyte", trainSize, 1)
    # trainLabels0是训练集数字图像对应的label
    trainLabels0 = extractData(r"..\data\knn\train-labels.idx1-ubyte", trainSize, 2)
    # testMat0是测试集数字图像数据数组
    testMat0 = extractData(r"..\data\knn\t10k-images.idx3-ubyte", testSize, 1)
    # testLabels0是测试集数字图像对应的label
    testLabels0 = extractData(r"..\data\knn\t10k-labels.idx1-ubyte", testSize, 2)

    trainMat1 = cutTransform(trainMat0, 7)
    testMat1 = cutTransform(testMat0, 7)

    print(trainMat1.shape)
    print(testMat1.shape)
    print(trainLabels0.shape)
    print(testLabels0.shape)

    # accuracy={}
    # for dim in range(2,30):
    #     w_vector = mycf.myNewLDA(trainMat1, trainLabels0, dim)
    #     x_train = np.dot(trainMat1, w_vector)
    #     x_test = np.dot(testMat1, w_vector)
    #     label_ = list(set(trainLabels0))
    #     x_classify = {}
    #
    #     for label in label_:
    #         xi = np.array([x_train[i] for i in range(len(x_train)) if trainLabels0[i] == label])
    #         x_classify[label] = xi
    #     # perceptron
    #     # thetaSet = {}
    #     # n = 0
    #     # for i in range(9):
    #     #     for j in range(i + 1, 10):
    #     #         thetaSet[(i, j)] = mycf.perceptron(x_classify[i], x_classify[j])
    #     # #         t = thetaSet[(i, j)]
    #     # #         print(t)
    #     # #         n += 1
    #     # #         plt.figure(n)
    #     # #         plt.scatter(x_classify[i][:, 0], x_classify[i][:, 1], marker='o')
    #     # #         plt.scatter(x_classify[j][:, 0], x_classify[j][:, 1], marker='*')
    #     # #         plt.plot([-0.1, 0.1], [-(t[0] - 0.1 * t[1]) / t[2], -(t[0] + 0.1 * t[1]) / t[2]])
    #     # # plt.show()
    #     # # print(len(thetaSet))
    #     # perceptron_predict = np.zeros((testLabels0.shape[0]))
    #     # for i in range(x_test.shape[0]):
    #     #     vote = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #     #     for key, value in thetaSet.items():
    #     #         # print(key)
    #     #         # print(len(value))
    #     #         re = np.dot(x_test[i], value[1:]) + value[0]
    #     #         # print(re)
    #     #         if Decimal(str(re)) >= Decimal('0.0'):
    #     #             vote[key[0]] += 1
    #     #         else:
    #     #             vote[key[1]] += 1
    #     #     m = 0
    #     #     for j in range(10):
    #     #         if (vote[j] > vote[m]):
    #     #             m = j
    #     #     perceptron_predict[i] = m
    #     # print(dim)
    #     # print(accuracyCount(perceptron_predict, testLabels0))
    #
    #     # logistic regression
    #     thetaSet = {}
    #     n=0
    #     for i in range(9):
    #         for j in range(i + 1, 10):
    #             thetaSet[(i, j)] = mycf.logistic_regression(x_classify[i], x_classify[j])
    #     #         t = thetaSet[(i, j)]
    #     #         n+=1
    #     #         plt.figure(n)
    #     #         plt.scatter(x_classify[i][:, 0], x_classify[i][:, 1], marker='o')
    #     #         plt.scatter(x_classify[j][:, 0], x_classify[j][:, 1], marker='*')
    #     #         plt.plot([-0.02, 0.03], [-(t[0] - 0.02 * t[1]) / t[2], -(t[0] + 0.03 * t[1]) / t[2]])
    #     # plt.show()
    #     logistic_predict = np.zeros((testLabels0.shape[0]))
    #     for i in range(x_test.shape[0]):
    #         vote = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #         for key, value in thetaSet.items():
    #             # print(x_test[i].dot(value[1:]) + value[0])
    #             if Sigmoid(x_test[i].dot(value[1:]) + value[0]) >= 0.5:
    #                 vote[key[0]] += 1
    #             else:
    #                 vote[key[1]] += 1
    #         m = 0
    #         # print(vote)
    #         for j in range(10):
    #             if vote[j] > vote[m]:
    #                 m = j
    #         logistic_predict[i] = m
    #
    #     # print(logistic_predict)
    #     # print(testLabels0)
    #     print(dim)
    #     print(accuracyCount(logistic_predict, testLabels0))
    #     accuracy[dim]=accuracyCount(logistic_predict, testLabels0)*100.0
    #
    # plt.figure(1)
    # plt.xlabel("dim")
    # plt.ylabel("accuracy/%")
    # plt.plot(accuracy.keys(),accuracy.values(),'ro-',color='#4169E1', alpha=0.8)
    # plt.show()

    # # least squares with regularization
    # # L1
    # lar = Lasso(alpha=0.5)
    # lar.fit(trainMat1, trainLabels0)
    # lar_y_predict = lar.predict(testMat1)
    # # print(lar_y_predict-testLabels0)
    # print(accuracyCount(lar_y_predict, testLabels0))
    #
    # # L2
    # myrr = myre.RidgeRegression(0.5)
    # myrr.fit(trainMat1, trainLabels0)
    # myrr_y_predict = myrr.predict(testMat1)
    # print(accuracyCount(myrr_y_predict, testLabels0))
    #
    # # Fisher discriminant analysis
    # # sklearn
    # lda = LinearDiscriminantAnalysis()
    # lda.fit(trainMat1, trainLabels0)
    # lda_y_predict = lda.predict(testMat1)
    # print(accuracyCount(lda_y_predict, testLabels0))
    #
    # # myimpl
    # w, cf_values = mycf.myLDA(trainMat1, trainLabels0, 9)
    # print(cf_values)
    # temp = np.dot(testMat1, w)
    # mylda_y_predict = np.zeros(testLabels0.shape[0])
    # for i in range(temp.shape[0]):
    #     for j in range(len(temp[i])):
    #         temp[i][j] = 1 if temp[i][j] > 0 else 0
    # for i in range(temp.shape[0]):
    #     equalCount = 0
    #     for k in cf_values.keys():
    #         tec = 0
    #         for j in range(len(cf_values[k])):
    #             tec += 1 if temp[i][j] == cf_values[k][j] else 0
    #         if tec > equalCount:
    #             equalCount = tec
    #             mylda_y_predict[i] = k
    # print(accuracyCount(mylda_y_predict, testLabels0))

    # 一个测试
    # t = mycf.logistic_regression(x_classify[0], x_classify[1])
    # plt.figure(1)
    # plt.scatter(x_classify[0][:, 0], x_classify[0][:, 1], marker='o')
    # plt.scatter(x_classify[1][:, 0], x_classify[1][:, 1], marker='*')
    # plt.plot([-0.1, 0.1], [-(t[0] - 0.1 * t[1]) / t[2], -(t[0] + 0.1 * t[1]) / t[2]])
    # plt.show()

    # LDA with kernel
    for dim in range(8,9):

        my_lda_with_kernel=LDAWithKernel()
        my_lda_with_kernel.fit(trainMat1,trainLabels0,dim)
        mylda_y_predict=my_lda_with_kernel.predict(testMat1)

        print("dim: %d",dim)
        print(accuracyCount(mylda_y_predict, testLabels0))

    # svm
    # linear
    # my_linear_svm_with_linear_kernel=svm.SVC(kernel='linear')
    # my_linear_svm_with_linear_kernel.fit(trainMat1,trainLabels0)
    #
    # linear_svm_predict_with_linear_kernel=my_linear_svm_with_linear_kernel.predict(testMat1)
    # print(accuracyCount(linear_svm_predict_with_linear_kernel,testLabels0))
    #
    # my_linear_svm_with_polynomial_kernel = svm.SVC(kernel='poly')
    # my_linear_svm_with_polynomial_kernel.fit(trainMat1, trainLabels0)
    #
    # linear_svm_predict_with_polynomial_kernel = my_linear_svm_with_polynomial_kernel.predict(testMat1)
    # print(accuracyCount(linear_svm_predict_with_polynomial_kernel, testLabels0))
    #
    # # nonlinear
    # my_nonlinear_svm = svm.NuSVC()
    # my_nonlinear_svm.fit(trainMat1, trainLabels0)
    #
    # nonlinear_svm_predict = my_nonlinear_svm.predict(testMat1)
    # print(accuracyCount(nonlinear_svm_predict, testLabels0))

handwritingClassfy(0.02)
