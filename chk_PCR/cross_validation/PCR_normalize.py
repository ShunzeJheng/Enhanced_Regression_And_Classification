# coding: utf-8

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import SCORERS


week_num = ['1-1', '1-2', '1-3', '1-4', '1-5', '2-3', '2-4', '2-5', '3-4', '3-5', '4-5']
week_tmp = ['2-5']
features_end_index = 23
total_features = 22
for week in week_tmp:
    datasets = pd.read_csv('../all_data_a/ncu_data_week_' + week + '.csv', sep=',')

    features_begin_index = 2
    #features_end_index += 1

    features_header = list(datasets)[features_begin_index : features_end_index + 1]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values

    number_of_folds = 10
    #total_features += 1
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)

    number_of_cv_evaluation = 100

    metrics_list = []

    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        kfold = KFold(n_splits=number_of_folds, shuffle=True)
        kfold_split_num = 1

        for train_index, test_index in kfold.split(features_val):
            features_val_train, features_val_test = features_val[train_index], features_val[test_index]
            label_val_train, label_val_test = label_val[train_index], label_val[test_index]


            for number_of_comp in range(1, total_features + 1):
                pca = PCA(n_components=number_of_comp)
                pca.fit(features_val_train)
                features_pca_val_train = pca.transform(features_val_train)
                MLR = linear_model.LinearRegression()
                MLR.fit(features_pca_val_train, label_val_train)
                features_pca_val_test = pca.transform(features_val_test)
                label_val_predict = MLR.predict(features_pca_val_test)
                #處理預測值超過不合理範圍
                for i in range(len(label_val_predict)):
                    if label_val_predict[i] > 100.00:
                        label_val_predict[i] = 100.00
                    elif label_val_predict[i] < 0:
                        label_val_predict[i] = 0.0
                #處理預測值超過不合理範圍

                train_scores , test_scores = [] , []
                scorer = SCORERS['r2']
                train_scores.append(scorer(MLR, features_pca_val_train, label_val_train))
                test_scores.append(scorer(MLR, features_pca_val_test, label_val_test))

                mean_train_score = np.mean(train_scores)
                mean_test_score = np.mean(test_scores)

                print (week + "_ mean_train_score : " + str(mean_train_score))
                print (week + "_ mean_test_score : " + str(mean_test_score))

                MAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
                #MSE = np.mean((label_val_predict - label_val_test) ** 2)
                MSE = sum((label_val_predict - label_val_test) ** 2)
                metrics_list.append([evaluation_num, kfold_split_num, number_of_comp, MAPC, MSE , mean_train_score , mean_test_score])
            #metrics_list.append([evaluation_num, kfold_split_num, number_of_comp, MSE])

            kfold_split_num = kfold_split_num + 1

    metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE' , 'mean_train_score' , 'mean_test_score'])

    # metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
    metrics_dataframe = metrics_dataframe.groupby(['evaluation_num', 'number_of_comp'], as_index=False).sum()
    # metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
    # metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
    metrics_dataframe.to_csv('all_result_a/PCR_metrics_list_' + week + '_normalize.csv', index=False)

    datasets = pd.read_csv('all_result_a/PCR_metrics_list_' + week + '_normalize.csv', sep=',')
    headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE']
    values = datasets[headers].values
    num_of_sample = len(features_val)
    result = []
    for item in values:
        features_number_plus_one = item[2] + 1
        result.append([item[0], item[1], item[2], item[3] / 10, item[4] / (num_of_sample - features_number_plus_one)])

    metrics_dataframe = pd.DataFrame(result, columns=headers)
    metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
    metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
    metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
    metrics_dataframe.to_csv('all_result_a/PCR_metrics_list_' + week + '_normalize.csv', index=False)

