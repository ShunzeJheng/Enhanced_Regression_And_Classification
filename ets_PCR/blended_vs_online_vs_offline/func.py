# coding: utf-8
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm

def normalize(final_score):
    return final_score,0
    #之前有做transformation，後來沒有使用
    """
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(final_score)
    final_score = scaler.transform(final_score)
    return final_score, scaler
"""
def inverse(final_score, scaler):
    return final_score
    # 之前有做transformation，後來沒有使用
    #return scaler.inverse_transform(final_score)

# 執行PCR，並輸出每一次交叉驗證的結果，包括預測分數、實際分數、預測準確度
def PCR(features_val, label_val, total_features, user_val,output_path_acc, output_path_residual, output_path_data_with_predict, features_header):
    label_val_normal, scaler = normalize(label_val)
    number_of_folds = 10
    number_of_cv_evaluation = 100

    metrics_list = []
    data_with_predict = []
    residual_result = []
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)

    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        kfold = KFold(n_splits=number_of_folds, shuffle=True)
        kfold_split_num = 1

        for train_index, test_index in kfold.split(features_val):
            user_val_test = user_val[test_index]
            features_val_train, features_val_test = features_val[train_index], features_val[test_index]
            label_val_train, label_val_test = label_val[train_index], label_val[test_index]
            label_val_normal_train, label_val_normal_test = label_val_normal[train_index], label_val_normal[test_index]
            for number_of_comp in range(1, total_features + 1):
                pca = PCA(n_components=number_of_comp)
                pca.fit(features_val_train)
                features_pca_val_train = pca.transform(features_val_train)
                MLR = linear_model.LinearRegression()
                MLR.fit(features_pca_val_train, label_val_normal_train)
                features_pca_val_test = pca.transform(features_val_test)
                label_val_predict = MLR.predict(features_pca_val_test)
                label_val_predict = inverse(label_val_predict, scaler)
                #處理預測值超過不合理範圍
                for i in range(len(label_val_predict)):
                    if label_val_predict[i] > 104.16:
                        label_val_predict[i] = 104.16
                    elif label_val_predict[i] < 0:
                        label_val_predict[i] = 0.0
                #處理預測值超過不合理範圍

                pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
                #pMSE = np.mean((label_val_predict - label_val_test) ** 2)
                pMSE = sum((label_val_predict - label_val_test) ** 2)
                metrics_list.append([evaluation_num, kfold_split_num, number_of_comp, pMAPC, pMSE])

                for i in range(len(label_val_predict)):
                    residual_result.append([evaluation_num, kfold_split_num, number_of_comp, label_val_predict[i], label_val_test[i]])
                    tmp_list = []
                    tmp_list.append(evaluation_num)
                    tmp_list.append(kfold_split_num)
                    tmp_list.append(number_of_comp)
                    tmp_list.append(user_val_test[i])
                    for col in features_val_test[i]:
                        tmp_list.append(col)
                    tmp_list.append(label_val_predict[i])
                    tmp_list.append(label_val_test[i])
                    data_with_predict.append(tmp_list)

            kfold_split_num = kfold_split_num + 1



    metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE'])

    metrics_dataframe = metrics_dataframe.groupby(['evaluation_num', 'number_of_comp'], as_index=False).sum()
    #metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
    #metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
    #metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
    metrics_dataframe.to_csv(output_path_acc, index=False)

    metrics_dataframe = pd.DataFrame(residual_result, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'label_val_predict',
                                                               'label_val_test'])
    metrics_dataframe.to_csv(output_path_residual, index=False)

    datasets_features = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'username', 'label_val_predict', 'label_val_test']
    datasets_features[3:3] = features_header
    metrics_dataframe = pd.DataFrame(data_with_predict, columns=datasets_features)
    metrics_dataframe.to_csv(output_path_data_with_predict, index=False)

#輸出特徵係數的p-value與加入r square、adjusted r square
def OLS_Regression(output_path_acc, total_features, features_val, label_val, dataset_name):
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)
    r_2 = []
    r_2_adj = []
    for number_of_comp in range(1, total_features + 1):
        pca = PCA(n_components=number_of_comp)
        pca.fit(features_val)
        features_val_pca = pca.fit_transform(features_val)
        features_val_pca = sm.add_constant(features_val_pca) #sklearn 預設有加入截距，statsmodels沒有，所以要加
        result = sm.OLS(label_val, features_val_pca).fit()
        print()
        print dataset_name + ":"
        print(result.summary())
        print()
        r_2.append(result.rsquared)
        r_2_adj.append(result.rsquared_adj)
    datasets = pd.read_csv(output_path_acc, sep=',')
    datasets.insert(3, 'r2', r_2)
    datasets.insert(4, 'r2_adj', r_2_adj)
    datasets.to_csv(output_path_acc, index=False)

#輸出經過pca轉換後的資料集
def data_after_pca_output(datasets_small_name, datasets, features_header, output_path_after_pca, output_path_mlr_coef):
    username_val = datasets['username'].values
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values
    total_features = len(features_header)
    label_val_normal, scaler = normalize(label_val)

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val_standard = standard_scaler.transform(features_val)


    pca = PCA(n_components=total_features)
    features_val_standard = pca.fit_transform(features_val_standard)

    result_metrics = []
    for i in range(0, len(username_val)):
        tmp_list = []
        tmp_list.append(username_val[i])
        for col in features_val_standard[i]:
            tmp_list.append(col)
        tmp_list.append(label_val[i])
        result_metrics.append(tmp_list)

    datasets_features = ['username', 'final_score']
    datasets_features[1:1] = features_header
    metrics_dataframe = pd.DataFrame(result_metrics, columns=datasets_features)
    metrics_dataframe.to_csv(output_path_after_pca, index=False)
    comp_headers = []
    for i in range(1,total_features+1):
        comp_headers.append('comp' + str(i))
    MLR = linear_model.LinearRegression()
    MLR.fit(features_val_standard, label_val)
    label_val_predict = MLR.predict(features_val_standard)
    mse = np.mean((label_val_predict - label_val) ** 2)
    print mse
    print "MLR coef: "
    print MLR.coef_
    coef_result = []
    coef_result.append(MLR.coef_)

    metrics_dataframe = pd.DataFrame(coef_result, columns=comp_headers)
    metrics_dataframe.to_csv(output_path_mlr_coef, index=False)
    #test(pca,MLR)

    final_result = []
    for i in range(0,len(username_val)):
        tmp = []
        tmp.append(username_val[i])
        for col in features_val[i]:
            tmp.append(col)
        tmp.append(label_val_predict[i])
        tmp.append(label_val[i])
        final_result.append(tmp)
    datasets_features = ['username', 'predict_score', 'real_score']
    datasets_features[1:1] = features_header
    metrics_dataframe = pd.DataFrame(final_result, columns=datasets_features)
    metrics_dataframe.to_csv('C:/Users/kslab/PycharmProjects/ets_PCR/blended_vs_online_vs_offline/result/residual_' + datasets_small_name + '.csv', index=False)

#計算pMSE
def modified_mse(output_path_acc, data_num):
    datasets = pd.read_csv(output_path_acc, sep=',')
    headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE']
    #headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE', 'SS_Residual', 'SS_Total']
    #result_headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE', 'R_square']
    values = datasets[headers].values
    num_of_sample = data_num
    result = []
    for item in values:
        features_number_plus_one = item[2]
        result.append([item[0], item[1], item[2], item[3]/10, item[4]/(num_of_sample - features_number_plus_one)])

    metrics_dataframe = pd.DataFrame(result, columns=headers)
    metrics_dataframe = metrics_dataframe.groupby(['number_of_comp'], as_index=False).mean()
    metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
    metrics_dataframe = metrics_dataframe.drop('kfold_split_num', 1)
    metrics_dataframe.to_csv(output_path_acc, index=False)

#輸出標準化過後的資料集
def data_standard(user_val, features_val, label_val, output_path_data_standard, features_header):
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val_standard = standard_scaler.transform(features_val)

    result_metrics = []
    for i in range(0, len(user_val)):
        tmp_list = []
        tmp_list.append(user_val[i])
        for col in features_val_standard[i]:
            tmp_list.append(col)
        tmp_list.append(label_val[i])
        result_metrics.append(tmp_list)

    datasets_features = ['username', 'final_score']
    datasets_features[1:1] = features_header
    metrics_dataframe = pd.DataFrame(result_metrics, columns=datasets_features)
    metrics_dataframe.to_csv(output_path_data_standard, index=False)
