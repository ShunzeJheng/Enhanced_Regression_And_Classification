# coding: utf-8
#程式目的:輸出高三增能班資料集根據pearson係數與顯著性進行特徵選取的結果，並輸出資料集

import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd
from scipy.spatial.distance import correlation
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

filename_list = ["1-6"]
filename_list = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6']
for filename in filename_list:
    dataset = pd.read_csv('chkb_data/ncu_data_week_' + filename + '.csv', sep=',')
    pearon_dataset = pd.read_csv('chkb_result/correlation_' + filename + '.csv', sep=',')
    feature_list = pearon_dataset['feature'].values
    pvalue_list = pearon_dataset['p-value'].values

    for i in range(0,len(feature_list)-1):
        final_feature = ['username']
        for j in range(0,len(pvalue_list)):
            if pvalue_list[j] < 0.05:
                final_feature.append(feature_list[j])
        final_feature.append('final_score')
        metrics = dataset[final_feature]
        metrics.to_csv('chkb_result/ncu_data_week_' + filename + '.csv', index=False)

