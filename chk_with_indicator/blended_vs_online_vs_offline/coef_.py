# coding: utf-8
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm


def get_coef(features_val, label_val, total_features):
    number_of_folds = 10
    number_of_cv_evaluation = 100

    metrics_list = []
    coef_list = []
    coef_list_count = 0
    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        kfold = KFold(n_splits=number_of_folds, shuffle=True)
        kfold_split_num = 1

        for train_index, test_index in kfold.split(features_val):
            features_val_train, features_val_test = features_val[train_index], features_val[test_index]
            label_val_train, label_val_test = label_val[train_index], label_val[test_index]

            """
            standard_scaler = preprocessing.StandardScaler()
            standard_scaler.fit(features_val_train)
            features_val_train = standard_scaler.transform(features_val_train)
            features_val_test = standard_scaler.transform(features_val_test)
"""

            pca = PCA(n_components=total_features)
            pca.fit(features_val_train)
            features_pca_val_train = pca.transform(features_val_train)
            MLR = linear_model.LinearRegression()
            MLR.fit(features_pca_val_train, label_val_train)
            features_pca_val_test = pca.transform(features_val_test)
            label_val_predict = MLR.predict(features_pca_val_test)
            # 處理預測值超過不合理範圍
            for i in range(len(label_val_predict)):
                if label_val_predict[i] > 104.16:
                    label_val_predict[i] = 104.16
                elif label_val_predict[i] < 0:
                    label_val_predict[i] = 0.0
            # 處理預測值超過不合理範圍
            #print MLR.coef_
            if len(coef_list) == 0:
                coef_list = MLR.coef_
                coef_list_count += 1
            else:
                coef_list += MLR.coef_
                coef_list_count += 1


            pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
            pMSE = np.mean((label_val_predict - label_val_test) ** 2)
            metrics_list.append([evaluation_num, kfold_split_num, total_features, pMAPC, pMSE])

            kfold_split_num = kfold_split_num + 1

    coef_result = coef_list/coef_list_count
    print "coef_mean" + str(coef_result)
    metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE'])

    metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
    metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
    metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
    metrics_dataframe.to_csv('result/pcr_mse_blended.csv', index=False)