import re
from functools import partial
from glob import glob

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
# from xfeat import (ArithmeticCombinations, ConcatCombination,
#                    GBDTFeatureExplorer, GBDTFeatureSelector, LabelEncoder,
#                    LambdaEncoder, Pipeline, SelectCategorical, SelectNumerical,
#                    TargetEncoder, aggregation)

# sns.set()
paths = glob(r'..\cyuko\data-chuko\train\*')
train_dfs = []
for path in paths:
    train_df = pd.read_csv(path)
    train_dfs.append(train_df)
train_df = pd.concat(train_dfs)
train_df.reset_index(drop=True, inplace=True)
print(train_df.head())
print(train_df.info)
# test_df = pd.read_csv('../data/input/test.csv')
# sub_df = pd.read_csv('../data/input/sample_submission.csv')

