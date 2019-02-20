# coding: utf-8
#輸出Ridge Regression結果

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import SCORERS
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

#指定特徵
features_begin_index = 1
features_end_index = 55
total_features = features_end_index - features_end_index +1

#進行模型訓練與評估
week_num = ['1-6(1a)', '1-12(2a)', '1-18(3a)', '7-12(2d)', '13-18(3d)']
week_test = ['1-18(3a)']
for week in week_test:
    # 讀取資料並指定特徵與LABEL
    datasets = pd.read_csv('data/ncu_data_week_' + week + '.csv', sep=',')

    features_header = list(datasets)[features_begin_index : features_end_index + 1]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values
    data_num = len(label_val)

    number_of_folds = 10
    number_of_cv_evaluation = 100

    metrics_list = []
    each_data_list = []

    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        kfold = KFold(n_splits=number_of_folds, shuffle=True)
        kfold_split_num = 1

        for train_index, test_index in kfold.split(features_val):
            features_val_train, features_val_test = features_val[train_index], features_val[test_index]
            label_val_train, label_val_test = label_val[train_index], label_val[test_index]

            # 標準化
            standard_scaler = preprocessing.StandardScaler()
            standard_scaler.fit(features_val_train)
            features_val_train = standard_scaler.transform(features_val_train)
            features_val_test = standard_scaler.transform(features_val_test)

            # regression model
            model = Ridge(alpha=1.0)
            model.fit(features_val_train, label_val_train)
            label_val_predict = model.predict(features_val_test)
            #處理預測值超過不合理範圍
            for i in range(len(label_val_predict)):
                if label_val_predict[i] > 104.16:
                    label_val_predict[i] = 104.16
                elif label_val_predict[i] < 0:
                    label_val_predict[i] = 0.0

            # 計算train_scores與test_scores，檢查是否overfitting
            train_scores , test_scores = [] , []
            scorer = SCORERS['r2']
            train_scores.append(scorer(model, features_val_train, label_val_train))
            test_scores.append(scorer(model, features_val_test, label_val_test))

            mean_train_score = np.mean(train_scores)
            mean_test_score = np.mean(test_scores)

            MAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
            MSE = sum((label_val_predict - label_val_test) ** 2)
            metrics_list.append([evaluation_num, kfold_split_num, MAPC, MSE , mean_train_score , mean_test_score])
            for i in range(len(label_val_predict)):
                each_data_list.append([evaluation_num, kfold_split_num, label_val_predict[i], label_val_test[i]])

            kfold_split_num = kfold_split_num + 1

    metrics_dataframe = pd.DataFrame(metrics_list,
                                 columns=['evaluation_num', 'kfold_split_num', 'pMAPC', 'pMSE',
                                          'mean_train_score', 'mean_test_score'])

    metrics_dataframe = metrics_dataframe.groupby(['evaluation_num'], as_index=False).sum()
    metrics_dataframe.to_csv('result/ridge/PCR_metrics_list_' + week + '_normalize.csv', index=False)

    metrics_dataframe = pd.DataFrame(each_data_list, columns=['evaluation_num', 'kfold_split_num', 'label_val_predict',
                                                              'label_val_test'])
    metrics_dataframe.to_csv('result/ridge/residual/cross_validation_Residual_table_' + week + '_normalize.csv', index=False)

    datasets = pd.read_csv('result/ridge/PCR_metrics_list_' + week + '_normalize.csv', sep=',')
    headers = ['evaluation_num', 'kfold_split_num', 'pMAPC', 'pMSE', 'mean_train_score', 'mean_test_score']
    values = datasets[headers].values
    num_of_sample = len(features_val)
    result = []
    for item in values:
        result.append([item[0], item[1], item[2] / 10, item[3] / (data_num - total_features), item[4] / 10, item[5] / 10])

    metrics_dataframe = pd.DataFrame(result, columns=headers)
    metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
    metrics_dataframe.to_csv('result/ridge/PCR_metrics_list_' + week + '_normalize.csv', index=False)

    # 計算各項指標
    datasets = pd.read_csv('result/ridge/residual/cross_validation_Residual_table_' + week + '_normalize.csv', sep=',')
    headers = list(datasets)[0: 4]
    metric = []
    for num in range(1, 101):
        data = datasets[datasets.evaluation_num == num]
        y_true = data['label_val_test'].values
        y_pred = data['label_val_predict'].values
        y_mean = np.mean(y_true)
        n = len(y_true)
        k = total_features
        ss_total = sum((y_true - y_mean) ** 2)
        ss_reg = sum((y_pred - y_mean) ** 2)
        mse = sum((y_pred - y_true) ** 2)
        pmse = mse / (n - k)
        # r_square = float(ss_reg)/float(ss_total)
        r_square = r2_score(y_true, y_pred)
        adj_r_square = 1.0 - ((n - 1.0) * (1.0 - r_square) / (n - k - 1.0))
        metric.append([num, pmse, r_square, adj_r_square])
    metrics_dataframe = pd.DataFrame(metric, columns=['evaluation_num', 'pmse', 'r_square', 'adj_r_square'])
    metrics_dataframe.to_csv('result/ridge/performance_' + week + '.csv', index=False)