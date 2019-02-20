# coding: utf-8
#程式目的:輸出地科班資料集根據single variable中每個特徵的準確度進行特徵選取的結果，並輸出資料集

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

filename_list = ["1-18(3a)"]
for filename in filename_list:
    dataset = pd.read_csv('ets_data/ncu_data_week_' + filename + '.csv', sep=',')
    auc_dataset = pd.read_csv('ets_result/classifier_' + filename + '.csv', sep=',')
    feature_list = auc_dataset['feature'].values
    algorithm_header = ['logistic', 'linear_svc', 'svc', 'GaNB', 'Decision_tree', 'Random_forest', 'NN']
    result_folder = ['logistic', 'linear_svc', 'svc', 'GaNB', 'dt', 'rf', 'nn']

    for i in range(0,len(algorithm_header)):
        final_feature = ['username']
        auc = auc_dataset[algorithm_header[i]]
        for j in range(0,len(auc)):
            if auc[j] > 0.5:
                final_feature.append(feature_list[j])
        final_feature.append('final_score')
        metrics = dataset[final_feature]
        metrics.to_csv('ets_result/' + result_folder[i] + '/ncu_data_week_' + filename + '.csv', index=False)

