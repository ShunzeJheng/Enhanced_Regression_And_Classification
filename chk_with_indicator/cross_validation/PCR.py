# coding: utf-8
#輸出資料集在各個component的表現

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def cross_validation(file, features_begin_index, features_end_index, num_of_cluster):

    filename_list = [file]
    print filename_list
    for filename in filename_list:
        # 讀取資料並指定特徵與LABEL
        datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/all_data_a/ncu_data_week_' + filename + '.csv', sep=',')


        features_header = list(datasets)[features_begin_index : features_end_index + 1]
        features_header_with_indicator = list(datasets)[features_begin_index : features_end_index + 2]
        features_val = datasets[features_header].values

        #判斷有幾個indicator，並加入header
        indicator_header = []
        if num_of_cluster == 2:
            indicator_header.append('indicator')
        else:
            for i in range(1,num_of_cluster):
                indicator_header.append('indicator' + str(i))

        indicator_val = datasets[indicator_header].values
        label_header = 'final_score'
        label_val = datasets[label_header].values

        number_of_folds = 10
        total_features = features_end_index - features_begin_index + 1
        number_of_cv_evaluation = 100

        # 標準化
        standard_scaler = preprocessing.StandardScaler()
        standard_scaler.fit(features_val)
        features_val = standard_scaler.transform(features_val)

        # 輸出各個component的表現
        metrics_list = []

        for evaluation_num in range(1, number_of_cv_evaluation + 1):
            kfold = KFold(n_splits=number_of_folds, shuffle=True)
            kfold_split_num = 1

            for train_index, test_index in kfold.split(features_val):
                features_val_train, features_val_test = features_val[train_index], features_val[test_index]
                indicator_val_train, indicator_val_test = indicator_val[train_index], indicator_val[test_index]
                label_val_train, label_val_test = label_val[train_index], label_val[test_index]

                for number_of_comp in range(1, total_features + 1):
                    pca = PCA(n_components=number_of_comp)
                    pca.fit(features_val_train)
                    features_pca_val_train = pca.transform(features_val_train)
                    pca_header_with_indicator = []
                    for i in range(0,number_of_comp):
                        pca_header_with_indicator.append('comp' + str(i+1))
                    if num_of_cluster == 2:
                        pca_header_with_indicator.append('indicator')
                    else:
                        for i in range(1, num_of_cluster):
                            pca_header_with_indicator.append('indicator' + str(i))
                    merge_metrics = []
                    for i in range(0, len(features_pca_val_train)):
                        tmp = []
                        for col in features_pca_val_train[i]:
                            tmp.append(col)
                        for val in indicator_val_train[i]:
                            tmp.append(val)
                        merge_metrics.append(tmp)

                    features_pca_val_train = pd.DataFrame(merge_metrics, columns=pca_header_with_indicator)
                    features_pca_val_train = features_pca_val_train.values

                    MLR = linear_model.LinearRegression()
                    MLR.fit(features_pca_val_train, label_val_train)

                    features_pca_val_test = pca.transform(features_val_test)

                    merge_metrics = []
                    for i in range(0, len(features_pca_val_test)):
                        tmp = []
                        for col in features_pca_val_test[i]:
                            tmp.append(col)
                        for val in indicator_val_test[i]:
                            tmp.append(val)
                        merge_metrics.append(tmp)

                    features_pca_val_test = pd.DataFrame(merge_metrics, columns=pca_header_with_indicator)
                    features_pca_val_test = features_pca_val_test.values

                    label_val_predict = MLR.predict(features_pca_val_test)
                    #處理預測值超過不合理範圍
                    for i in range(len(label_val_predict)):
                        if label_val_predict[i] > 104.16:
                            label_val_predict[i] = 104.16
                        elif label_val_predict[i] < 0:
                            label_val_predict[i] = 0.0
                    #處理預測值超過不合理範圍


                    MAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
                    MSE = sum((label_val_predict - label_val_test) ** 2)
                    metrics_list.append([evaluation_num, kfold_split_num, number_of_comp, MAPC, MSE])
                kfold_split_num = kfold_split_num + 1

        metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE'])
        metrics_dataframe = metrics_dataframe.groupby(['evaluation_num', 'number_of_comp'], as_index=False).sum()
        metrics_dataframe.to_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/cross_validation/all_result_a/PCR_metrics_list_' + file + '_normalize.csv', index=False)

        # 計算各項指標
        datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/cross_validation/all_result_a/PCR_metrics_list_' + file + '_normalize.csv', sep=',')
        headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE']
        values = datasets[headers].values
        num_of_sample = len(features_val)
        result = []
        for item in values:
            features_number_plus_one = item[2]
            result.append([item[0], item[1], item[2], item[3] / 10, item[4] / (num_of_sample - features_number_plus_one)])

        metrics_dataframe = pd.DataFrame(result, columns=headers)
        metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
        metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
        metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
        metrics_dataframe.to_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/cross_validation/all_result_a/PCR_metrics_list_' + file + '_normalize.csv', index=False)


