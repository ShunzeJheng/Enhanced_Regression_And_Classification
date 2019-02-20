# coding: utf-8
#程式目的:輸出地科班資料集根據pearson係數與顯著性進行特徵選取的結果，並輸出資料集

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

filename_list = ['1-6(1a)', '1-12(2a)', '1-18(3a)', '7-12(2d)', '13-18(3d)']
for filename in filename_list:
    dataset = pd.read_csv('ets_data/ncu_data_week_' + filename + '.csv', sep=',')
    pearon_dataset = pd.read_csv('ets_result/correlation_' + filename + '.csv', sep=',')
    feature_list = pearon_dataset['feature'].values
    pvalue_list = pearon_dataset['p-value'].values

    for i in range(0,len(feature_list)-1):
        final_feature = ['username']
        for j in range(0,len(pvalue_list)):
            if pvalue_list[j] < 0.05:
                final_feature.append(feature_list[j])
        final_feature.append('final_score')
        metrics = dataset[final_feature]
        metrics.to_csv('ets_result/ncu_data_week_' + filename + '.csv', index=False)

