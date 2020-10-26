from numpy import *
import operator
from math import *


# 从文件读取数据
def extractData(fileName, size, type):
    file = open(fileName, "rb")
    if type == 1:
        file.seek(16)
        dataMat = zeros((size, 784))
        for i in range(size):
            for j in range(28 * 28):
                pixel = file.read(1)
                dataMat[i][j] = ord(pixel) * 1.0 / 255
    else:
        file.seek(8)
        dataMat = []
        for i in range(size):
            dataMat.append(ord(file.read(1)))
    file.close()
    return dataMat


# sweep变换提取特征
def sweepTransform(dataMat):
    size = dataMat.shape[0]
    sq = int(sqrt(dataMat.shape[1]))
    lessSize = int(sq * 1.6)
    while lessSize % 4:
        lessSize -= 1
    # print(lessSize)
    # print(sq)
    dataMatAdvance = zeros((size, lessSize))
    for i in range(size):
        pixelSum = 0.0
        xAxis = 0.0
        yAxis = 0.0
        for r in range(sq):
            for c in range(sq):
                pixelSum += dataMat[i][r * sq + c]
                xAxis += r * dataMat[i][r * sq + c]
                yAxis += c * dataMat[i][r * sq + c]
        xAxis /= pixelSum
        yAxis /= pixelSum
        x = int(xAxis)
        y = int(yAxis)
        # print(yAxis)
        for r in range(sq):
            for c in range(sq):
                if r > x - lessSize / 4 and r <= lessSize / 4:
                    dataMatAdvance[i][r - x + int(lessSize / 4) - 1] += dataMat[i][r * sq + c] * (r - xAxis)
                if c > y - lessSize / 4 and c <= lessSize / 4:
                    dataMatAdvance[i][c - y + int(lessSize * 3 / 4) - 1] += dataMat[i][r * sq + c] * (c - yAxis)

        # print(dataMatAdvance)
    return dataMatAdvance


# cut变换提取特征
def cutTransform(dataMat, div):
    size = dataMat.shape[0]
    dataMatAdvance = zeros((size, div * div))
    l = 28 / div
    for i in range(size):
        for r in range(28):
            for c in range(28):
                dataMatAdvance[i][int(r / l) * div + int(c / l)] += dataMat[i][r * 28 + c]

    dataMatAdvance /= l * l
    return dataMatAdvance


# 执行knn算法并打印结果
def executeAndPrint(trainMat, trainLabels, testMat, testLabels, k):
    testSize = testMat.shape[0]
    errorCount = 0
    for index in range(testSize):
        # knnClassify是用于分类的knn算法
        classifyRes = knnClassify(trainMat, trainLabels, testMat[index], k)
        # print(expectLabel,end=" ")
        # print(classifyRes)
        errorCount += classifyRes != testLabels[index]

    print("k=%d" % k)
    print("testSize: %d" % testSize)
    print("errorRate: %f%%" % (errorCount * 100.0 / testSize))


def handwritingClassfy(percent, k):
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

    index = 1
    print("Algorithm%d" % index)
    index += 1
    # 原始数据执行knn算法
    executeAndPrint(trainMat0, trainLabels0, testMat0, testLabels0, k)

    trainMat1 = sweepTransform(trainMat0)
    testMat1 = sweepTransform(testMat0)
    print("Algorithm%d" % index)
    index += 1
    # sweep变换提取特征后执行knn算法
    executeAndPrint(trainMat1, trainLabels0, testMat1, testLabels0, k)

    trainMat2 = cutTransform(trainMat0, 7)
    testMat2 = cutTransform(testMat0, 7)
    print("Algorithm%d" % index)
    index += 1
    # cut变换提取特征后执行knn算法，将28*28变成7*7
    executeAndPrint(trainMat2, trainLabels0, testMat2, testLabels0, k)

    trainMat3 = cutTransform(trainMat0, 14)
    testMat3 = cutTransform(testMat0, 14)
    print("Algorithm%d" % index)
    index += 1
    # cut变换提取特征后执行knn算法，将28*28变成14*14
    executeAndPrint(trainMat3, trainLabels0, testMat3, testLabels0, k)


def knnClassify(trainFeature, trainLabel, testFeature, k):
    elementNum = trainFeature.shape[0]
    diffMat = tile(testFeature, (elementNum, 1)) - trainFeature
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = trainLabel[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


handwritingClassfy(0.1, 2)
handwritingClassfy(0.1, 3)
handwritingClassfy(0.1, 4)
handwritingClassfy(0.1, 5)
handwritingClassfy(0.1, 7)
handwritingClassfy(0.1, 10)
handwritingClassfy(0.1, 15)
handwritingClassfy(0.1, 20)
