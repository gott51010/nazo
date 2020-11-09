import pandas as pd
# 混淆矩阵
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

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

# 混淆矩阵（Confusion Matrix）:
# 混淆矩阵的每一列代表了预测类别，每一列的总数表示预测为该类别的数据的数目；每一行代表了数据的真实归属类别，
# 把正例正确分类为正例，表示为TP（true positive），把正例错误分类为负例，表示为FN（false negative），
# 把负例正确分类为负例，表示为TN（true negative）， 把负例错误分类为正例，表示为FP（false positive）
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]
print('混淆矩阵:\n', confusion_matrix(y_true, y_pred))


y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]
print('ACC:',accuracy_score(y_true, y_pred))


## Precision,Recall,F1-score
from sklearn import metrics
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]
#  precision = TP/(TP + FP),  recall = TP/(TP +FＮ)
print('Precision',metrics.precision_score(y_true, y_pred))
print('Recall',metrics.recall_score(y_true, y_pred))
print('F1-score:',metrics.f1_score(y_true, y_pred))


## P-R曲线
# P-R曲线就是精确率precision vs 召回率recall 曲线，以recall作为横坐标轴，precision作为纵坐标轴。
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
plt.plot(precision, recall)
plt.show()

## ROC曲线
from sklearn.metrics import roc_curve
y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
FPR,TPR,thresholds=roc_curve(y_true, y_pred)
plt.title('ROC')
plt.plot(FPR, TPR,'b')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

## AUC
# AUC（Area under roc Curve）面积，这个概念其实很简单，就是指ROC曲线下的面积大小，
# 而计算AUC值只需要沿着ROC横轴做积分就可以了。真实场景中ROC曲线一般都会在这条直线的上方，
# 所以AUC的取值一般在0.5~1之间。AUC的值越大，说明该模型的性能越好。
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('AUC socre:',roc_auc_score(y_true, y_scores))

## KS值 在实际操作时往往使用ROC曲线配合求出KS值
from sklearn.metrics import roc_curve
y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
FPR,TPR,thresholds=roc_curve(y_true, y_pred)
KS=abs(FPR-TPR).max()
print('KS值：',KS)

#评分卡
def Score(prob,P0=600,PDO=20,badrate=None,goodrate=None):
    P0 = P0
    PDO = PDO
    theta0 = badrate/goodrate
    B = PDO/np.log(2)
    A = P0 + B*np.log(2*theta0)
    score = A-B*np.log(prob/(1-prob))
    return score
