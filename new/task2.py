import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# seaborn 是在matplotlib 的基础上的高级封装
import seaborn as sns
import datetime
import warnings

warnings.filterwarnings('ignore')
import os
import pandas.util.testing as tm

# pandas读取数据时相对路径载入报错时，尝试使用os.getcwd()查看当前工作目录。
# print(os.getcwd())

# TSV与CSV的区别：
# a. 从名称上即可知道，TSV是用制表符（Tab,'\t'）作为字段值的分隔符；CSV是用半角逗号（','）作为字段
# 值的分隔符；
# b. Python对TSV文件的支持：
# Python的csv模块准确的讲应该叫做dsv模块，因为它实际上是支持范式的分隔符分隔值文件（DSV，
# delimiter-separated values）的。
# delimiter参数值默认为半角逗号，即默认将被处理文件视为CSV。当delimiter='\t'时，被处理文件就是TSV。


# 设置chunksize参数，来控制每次迭代数据的大小
# chunker = pd.read_csv("data/train.csv", chunksize=5)
# for item in chunker:
#     print(type(item))
#     print(len(item))

# 读取 部分数据 nrow 参数
data_train = pd.read_csv('data/train.csv')
data_test_a = pd.read_csv('data/testA.csv')
# 观察数据
# print(data_train.shape)
# print(data_test_a.shape)
print(data_train.columns)
print(data_train.info())
# 查看各个特征的统计量
print(data_train.describe())
print("统计量的首尾数据确认")
print(data_train.head(3).append(data_train.tail(3)))

# print(data_train.isnull().any())
# 查看缺失值
print(f'There are {data_train.isnull().any().sum()} columns in train dataset with missing values.')
# There are 22 columns in train dataset with missing values.
# 可见有22列特征是有缺失的

# 进一步查看缺失特征中缺失率大于50%的特征

have_null_fea_dict = (data_train.isnull().sum()/len(data_train)).to_dict()
# print(have_null_fea_dict)
fea_null_moreThanHalf = {}
for key,value in have_null_fea_dict.items():
    if value > 0.5:
        fea_null_moreThanHalf[key] = value
        
# print(have_null_fea_dict)
# {} 哟西,并没有确实一半以上的数据


# 具体的查看缺失特征及缺失率 把 nan数据可视化
# missing = data_train.isnull().sum()/len(data_train)
# missing = missing[missing > 0]
# missing.sort_values(inplace=True)
# missing.plot.bar()
# plt.show()

# 如果缺失nan值真的很大 可以考虑删掉 ,如果很小可以考虑填充,或者结合模型自行优化


# 查看训练集测试集中特征属性只有一值的特征
one_value_fea = [col for col in data_train.columns if data_train[col].nunique() <= 1]
print("只有一值的特征",one_value_fea)  # ['policyCode'] 公开可用的策略代码=1新产品不公开可用的策略代码=2

# ===========================截至目前为止的综合分析==============================
# 47列数据中有22列都缺少数据，‘policyCode’具有一个唯一值（或全部缺失）。有很多连续变量和一些分类变量。






