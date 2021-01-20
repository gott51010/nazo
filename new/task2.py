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

# 查看特征的数值类型有哪些，对象类型有哪些
numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
print(numerical_fea)
print(data_train.grade)

#过滤数值型类别特征
def get_numerical_serial_fea(data, feas):
    numerical_serial_fea = []
    numerical_noserial_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        if temp <= 10:
            numerical_noserial_fea.append(fea)
            continue
        numerical_serial_fea.append(fea)
    return numerical_serial_fea, numerical_noserial_fea

numerical_serial_fea, numerical_noserial_fea = get_numerical_serial_fea(data_train, numerical_fea)
# 值型变量分析，数值型肯定是包括连续型变量和离散型变量的，找出来
print(numerical_serial_fea)
print(data_train['term'].value_counts(),"离散型变量")
print(data_train['homeOwnership'].value_counts(),"离散型变量")
print(data_train['verificationStatus'].value_counts(),"离散型变量")
print(data_train['policyCode'].value_counts(),"离散型变量，无用，全部一个值")
print(data_train['n11'].value_counts(),"离散型变量，相差悬殊，用不用再分析")
print(data_train['n12'].value_counts(),"离散型变量，相差悬殊，用不用再分析")

# 单一变量分布可视化
# plt.figure(figsize=(8, 8))
# sns.barplot(data_train["employmentLength"].value_counts(dropna=False)[:20],
# data_train["employmentLength"].value_counts(dropna=False).keys()[:20])
# plt.show()

# 时间格式数据处理及查看
#转化成时间格式
data_train['issueDate'] = pd.to_datetime(data_train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_train['issueDateDT'] = data_train['issueDate'].apply(lambda x: x-startdate).dt.days

data_test_a['issueDate'] = pd.to_datetime(data_train['issueDate'],format='%Y-%m-%d')
startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
data_test_a['issueDateDT'] = data_test_a['issueDate'].apply(lambda x: x-startdate).dt.days

plt.hist(data_train['issueDateDT'], label='train')
plt.hist(data_test_a['issueDateDT'], label='test')
plt.legend()
plt.title('Distribution of issueDateDT dates')
#train 和 test issueDateDT 日期有重叠 所以使用基于时间的分割进行验证是不明智的

import pandas_profiling
pfr = pandas_profiling.ProfileReport(data_train)
pfr.to_file("./example.html")