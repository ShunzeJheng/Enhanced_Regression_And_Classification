# coding: utf-8
#輸出資料集在各個component的表現
#在產生a班或b班的資料前，記得改資料夾或資料集的名稱，如class_a_csv或class_b_csv

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import boxcox

def transformation(final_score):
    return final_score,0
    #做BOXCOX轉換，之後沒有用到
    """
    if 0 in final_score:
        final_score = final_score + 1
    result = boxcox(final_score)
    final_score = result[0]
    lamda = result[1]
    return final_score,lamda
"""

def transformation_reverse(final_score, lamda):
    return final_score
    ##做BOXCOX反轉換，之後沒有用到
    """
    reverse = (final_score * lamda + 1) ** (1.0 / lamda)
    reverse = reverse - 1
    return reverse
"""


def cross_validation(file, features_begin_index, features_end_index):
    filename_list = [file]
    for filename in filename_list:
        #讀取資料並指定特徵與LABEL
        datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_PCR/all_data_a/ncu_data_week_' + filename + '.csv', sep=',')

        features_header = list(datasets)[features_begin_index : features_end_index + 1]
        features_val = datasets[features_header].values
        label_header = 'final_score'
        label_val = datasets[label_header].values

        label_val_transformation, transfor_lamda = transformation(label_val)

        number_of_folds = 10
        total_features = features_end_index - features_begin_index + 1

        # 標準化
        standard_scaler = preprocessing.StandardScaler()
        standard_scaler.fit(features_val)
        features_val = standard_scaler.transform(features_val)

        # 輸出各個component的表現
        number_of_cv_evaluation = 100
        metrics_list = []

        for evaluation_num in range(1, number_of_cv_evaluation + 1):
            kfold = KFold(n_splits=number_of_folds, shuffle=True)
            kfold_split_num = 1

            for train_index, test_index in kfold.split(features_val):
                features_val_train, features_val_test = features_val[train_index], features_val[test_index]
                label_val_train, label_val_test = label_val[train_index], label_val[test_index]
                label_val_transformation_train, label_val_transformation_test = label_val_transformation[train_index], label_val_transformation[test_index]

                for number_of_comp in range(1, total_features + 1):
                    pca = PCA(n_components=number_of_comp)
                    pca.fit(features_val_train)
                    features_pca_val_train = pca.transform(features_val_train)
                    MLR = linear_model.LinearRegression()
                    MLR.fit(features_pca_val_train, label_val_transformation_train)
                    features_pca_val_test = pca.transform(features_val_test)
                    label_val_predict = MLR.predict(features_pca_val_test)
                    label_val_predict = transformation_reverse(label_val_predict, transfor_lamda)

                    #處理預測值超過不合理範圍
                    for i in range(len(label_val_predict)):
                        if label_val_predict[i] > 100.00:
                            label_val_predict[i] = 100.00
                        elif label_val_predict[i] < 0:
                            label_val_predict[i] = 0.0

                    MAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
                    MSE = sum((label_val_predict - label_val_test) ** 2)
                    metrics_list.append([evaluation_num, kfold_split_num, number_of_comp, MAPC, MSE])
                kfold_split_num = kfold_split_num + 1

        metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE'])
        metrics_dataframe = metrics_dataframe.groupby(['evaluation_num', 'number_of_comp'], as_index=False).sum()
        metrics_dataframe.to_csv('C:/Users/kslab/PycharmProjects/chk_PCR/cross_validation/all_result_a/PCR_metrics_list_' + filename +'_normalize.csv', index=False)

        # 計算各項指標
        datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_PCR/cross_validation/all_result_a/PCR_metrics_list_' + filename +'_normalize.csv', sep=',')
        headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE']
        values = datasets[headers].values
        num_of_sample = len(features_val)
        result = []
        for item in values:
            features_number = item[2]
            result.append([item[0], item[1], item[2], item[3] / 10, item[4] / (num_of_sample - features_number)])

        metrics_dataframe = pd.DataFrame(result, columns=headers)
        metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
        metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
        metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
        metrics_dataframe.to_csv('C:/Users/kslab/PycharmProjects/chk_PCR/cross_validation/all_result_a/PCR_metrics_list_' + filename +'_normalize.csv', index=False)
