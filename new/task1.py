import pandas as pd
# 混淆矩阵
import numpy as np
from sklearn.metrics import confusion_matrix

# 1.2
# 赛题概况
# 比赛要求参赛选手根据给定的数据集，建立模型，预测金融风险。
# 赛题以预测金融风险为任务，数据集报名后可见并可下载，该数据来自某信贷平台的贷款记录，总数据量超过120w，
# 包含47列变量信息，其中15列为匿名变量。从中抽取80万条作为训练集，20 万条作为测试集A，20万条作为测试集B
# 同时会对employmentTitle、purpose、postCode和title等信息进行脱敏。


train = pd.read_csv('data/train.csv')
testA = pd.read_csv('data/testA.csv')

# # Train data shape: (800000, 47)
# print('Train data shape:',train.shape)
# # TestA data shape: (200000, 46)
# print('TestA data shape:',testA.shape)


print(train.head(10))
print(train.info())
print("查看缺失值", train.isnull().any())
