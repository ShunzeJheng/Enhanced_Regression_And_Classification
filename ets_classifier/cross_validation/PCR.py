# coding: utf-8

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing



filename_list = ["1-6(1a)", "1-12(2a)", "1-18(3a)", "7-12(2d)", "13-18(3d)"]
#filename_list = ["1-18(3a)"]

for filename in filename_list:
    datasets = pd.read_csv('../data/ncu_data_week_' + filename + '.csv', sep=',')
    """
    datasets = datasets[datasets.username != 102602012]
    datasets = datasets[datasets.username != 104602014]
    datasets = datasets[datasets.username != 104602026]
    datasets = datasets[datasets.username != 104602032]
    datasets = datasets[datasets.username != 104602017]
"""

    features_begin_index = 1
    features_end_index = 19

    features_header = list(datasets)[features_begin_index : features_end_index + 1]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values

    number_of_folds = 10
    total_features = features_end_index - features_begin_index + 1

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)


    number_of_cv_evaluation = 100

    metrics_list = []

    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        for number_of_comp in range(1, total_features + 1):
            pca = PCA(n_components=number_of_comp)
            pca.fit(features_val)
            features_pca_val = pca.transform(features_val)

            clf = linear_model.LogisticRegression()
            clf.fit(features_pca_val, label_val)
            label_val_predict = clf.predict(features_pca_val)

            error_num = 0.0
            for i in range(0,len(label_val)):
                if label_val[i] != label_val_predict[i]:
                    error_num = error_num + 1.0
            acc = (len(label_val) - error_num) / len(label_val)

            metrics_list.append([evaluation_num, number_of_comp, acc])


    metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'number_of_comp', 'acc'])
    metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
    metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
    metrics_dataframe.to_csv('result/PCA_logistic_' + filename + '.csv', index=False)