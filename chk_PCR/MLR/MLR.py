# coding: utf-8
#程式目的:產生MLR結果


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm

#讀取資料
datasets_small_name = '1-6'
datasets = pd.read_csv('../all_data_b/ncu_data_week_' + datasets_small_name + '.csv', sep=',')

#指定特徵與LABEL
features_begin_index = 1
features_end_index = 55
features_header = list(datasets)[features_begin_index : features_end_index + 1]
features_val = datasets[features_header].values
label_header = 'final_score'
label_val = datasets[label_header].values

number_of_folds = 10
total_features = features_end_index - features_begin_index + 1
number_of_cv_evaluation = 100

#標準化
standard_scaler = preprocessing.StandardScaler()
standard_scaler.fit(features_val)
features_val = standard_scaler.transform(features_val)

#進行MLR
metrics_list = []
coef_list = []
coef_list_count = 0
for evaluation_num in range(1, number_of_cv_evaluation + 1):
    kfold = KFold(n_splits=number_of_folds, shuffle=True)
    kfold_split_num = 1

    for train_index, test_index in kfold.split(features_val):
        features_val_train, features_val_test = features_val[train_index], features_val[test_index]
        label_val_train, label_val_test = label_val[train_index], label_val[test_index]

        features_val_train = sm.add_constant(features_val_train)
        result = sm.OLS(label_val_train, features_val_train).fit()
        features_val_test = sm.add_constant(features_val_test,has_constant='add')
        label_val_predict = result.predict(features_val_test)
        # 處理預測值超過不合理範圍
        for i in range(len(label_val_predict)):
            if label_val_predict[i] > 100.00:
                label_val_predict[i] = 100.00
            elif label_val_predict[i] < 0:
                label_val_predict[i] = 0.0

        if len(coef_list) == 0:
            coef_list = result.params
            coef_list_count += 1
        else:
            coef_list += result.params
            coef_list_count += 1

        pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
        pMSE = sum((label_val_predict - label_val_test) ** 2)
        metrics_list.append([evaluation_num, kfold_split_num, total_features, pMAPC, pMSE, result.rsquared, result.rsquared_adj])

        kfold_split_num = kfold_split_num + 1

coef_result = coef_list/coef_list_count
print "coef_mean" + str(coef_result)
metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE', 'r2', 'r2_adj'])
metrics_dataframe = metrics_dataframe.groupby(['evaluation_num', 'number_of_comp'], as_index=False).sum()
metrics_dataframe.to_csv('mlr_mse_blended_' + datasets_small_name +'_normalize.csv', index=False)

#計算各項指標
datasets = pd.read_csv('mlr_mse_blended_' + datasets_small_name +'_normalize.csv', sep=',')
headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE', 'r2', 'r2_adj']
values = datasets[headers].values
num_of_sample = len(features_val)
result = []
for item in values:
    result.append([item[0], item[1], item[2], item[3] / 10, item[4] / (num_of_sample - total_features), item[5] / 10, item[6] / 10])

metrics_dataframe = pd.DataFrame(result, columns=headers)
metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
metrics_dataframe.to_csv('mlr_mse_blended_' + datasets_small_name +'_normalize.csv', index=False)
