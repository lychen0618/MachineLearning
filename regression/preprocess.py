import csv
import numpy as np
import scipy.stats as sct


# 相关性分析
def pearson(x, y):
    cov = np.corrcoef(np.array(x), np.array(y))
    # print(cov)
    return True if abs(cov[0][1]) >= 0.5 else False


def spearman(x, y):
    scov = sct.spearmanr(np.array(x), np.array(y))[0]
    # print(scov)
    return True if abs(scov) >= 0.6 else False


def preprocess(fileName):
    dataArray = []
    with open(fileName, encoding="utf-8") as f:
        reader = csv.reader(f)
        for i in reader:
            temp = i[0].strip().split()
            dataArray.append(temp)
        # print(dataArray)

    dataArray = [[float(x) for x in row] for row in dataArray]  # 将数据从string形式转换为float形式
    dataArray = np.array(dataArray)  # 将list数组转化成array数组便于查看数据结构
    # print(dataArray.shape)  # 利用.shape查看结构
    priceColumn = dataArray.shape[1] - 1

    features = []
    for c in range(priceColumn):
        if pearson(dataArray[:, c], dataArray[:, priceColumn]) or spearman(dataArray[:, c], dataArray[:, priceColumn]):
            features.append(dataArray[:, c])
            print("column%d is selected" % c)

    target = np.array(dataArray[:, priceColumn])
    features = np.transpose(np.array(features))

    return features, target

# preprocess(r'..\data\regression\boston_house_prices.csv')
