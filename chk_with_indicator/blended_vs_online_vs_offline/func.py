# coding: utf-8
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing
import statsmodels.api as sm


# 執行PCR，並輸出每一次交叉驗證的結果，包括預測分數、實際分數、預測準確度
def PCR(features_val, label_val, total_features, user_val,output_path_acc, output_path_residual, output_path_data_with_predict, features_header,indicator_val, num_of_cluster):

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)

    number_of_folds = 10
    number_of_cv_evaluation = 100

    metrics_list = []
    data_with_predict = []
    residual_result = []


    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        kfold = KFold(n_splits=number_of_folds, shuffle=True)
        kfold_split_num = 1

        for train_index, test_index in kfold.split(features_val):
            user_val_test = user_val[test_index]
            features_val_train, features_val_test = features_val[train_index], features_val[test_index]
            indicator_val_train, indicator_val_test = indicator_val[train_index], indicator_val[test_index]
            label_val_train, label_val_test = label_val[train_index], label_val[test_index]


            for number_of_comp in range(1, total_features + 1):
                pca = PCA(n_components=number_of_comp)
                pca.fit(features_val_train)
                features_pca_val_train = pca.transform(features_val_train)

                pca_header_with_indicator = []
                for i in range(0, number_of_comp):
                    pca_header_with_indicator.append('comp' + str(i + 1))
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
                    for col in indicator_val_train[i]:
                        tmp.append(col)
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
                    for col in indicator_val_test[i]:
                        tmp.append(col)
                    merge_metrics.append(tmp)

                features_pca_val_test = pd.DataFrame(merge_metrics, columns=pca_header_with_indicator)
                features_pca_val_test = features_pca_val_test.values

                label_val_predict = MLR.predict(features_pca_val_test)
                #處理預測值超過不合理範圍
                for i in range(len(label_val_predict)):
                    if label_val_predict[i] > 100.00:
                        label_val_predict[i] = 100.00
                    elif label_val_predict[i] < 0:
                        label_val_predict[i] = 0.0
                #處理預測值超過不合理範圍

                pMAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / np.mean(label_val)))
                pMSE = sum((label_val_predict - label_val_test) ** 2)
                metrics_list.append([evaluation_num, kfold_split_num, number_of_comp, pMAPC, pMSE])
                if number_of_comp == total_features:
                    for i in range(len(label_val_predict)):
                        residual_result.append([evaluation_num, kfold_split_num, label_val_predict[i], label_val_test[i]])
                        tmp_list = []
                        tmp_list.append(evaluation_num)
                        tmp_list.append(kfold_split_num)
                        tmp_list.append(user_val_test[i])
                        for col in features_val_test[i]:
                            tmp_list.append(col)
                        tmp_list.append(label_val_predict[i])
                        tmp_list.append(label_val_test[i])
                        data_with_predict.append(tmp_list)

            kfold_split_num = kfold_split_num + 1



    metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE'])

    metrics_dataframe = metrics_dataframe.groupby(['evaluation_num', 'number_of_comp'], as_index=False).sum()
    metrics_dataframe.to_csv(output_path_acc, index=False)

    metrics_dataframe = pd.DataFrame(residual_result, columns=['evaluation_num', 'kfold_split_num', 'label_val_predict',
                                                               'label_val_test'])
    metrics_dataframe.to_csv(output_path_residual, index=False)

    datasets_features = ['evaluation_num', 'kfold_split_num', 'username', 'label_val_predict','label_val_test']
    datasets_features[3:3] = features_header
    metrics_dataframe = pd.DataFrame(data_with_predict, columns=datasets_features)
    metrics_dataframe.to_csv(output_path_data_with_predict, index=False)

#輸出特徵係數的p-value與加入r square、adjusted r square
def OLS_Regression(output_path_acc, file, total_features, features_val, label_val, dataset_name, indicator_val, num_of_cluster):
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)
    pca = PCA(n_components=total_features)
    features_val = pca.fit_transform(features_val)

    pca_header_with_indicator = []
    for i in range(0, total_features):
        pca_header_with_indicator.append('comp' + str(i + 1))
    if num_of_cluster == 2:
        pca_header_with_indicator.append('indicator')
    else:
        for i in range(1, num_of_cluster):
            pca_header_with_indicator.append('indicator' + str(i))

    merge_metrics = []
    for i in range(0, len(features_val)):
        tmp = []
        for col in features_val[i]:
            tmp.append(col)
        for col in indicator_val[i]:
            tmp.append(col)
        merge_metrics.append(tmp)
    features_val = pd.DataFrame(merge_metrics, columns=pca_header_with_indicator)
    features_val = features_val.values

    features_val = sm.add_constant(features_val) #sklearn 預設有加入截距，statsmodels沒有，所以要加
    result = sm.OLS(label_val, features_val).fit()
    print()
    print dataset_name + ":"
    print(result.summary())
    print()

    pvalue_l = []
    tmp = []
    for i in range(1, len(result.pvalues)):
        tmp.append(round(result.pvalues[i], 3))
    pvalue_l.append(tmp)
    variable_analysis_results_table = pd.DataFrame(pvalue_l, columns=pca_header_with_indicator)
    variable_analysis_results_table.to_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/blended_vs_online_vs_offline/all_result_a/pvalue_' + file + '.csv', index=False)
    r_2 = []
    r_2_adj = []
    for number_of_comp in range(1, total_features + 1):
        pca = PCA(n_components=number_of_comp)
        pca.fit(features_val)
        features_val_pca = pca.fit_transform(features_val)
        features_val_pca = sm.add_constant(features_val_pca)  # sklearn 預設有加入截距，statsmodels沒有，所以要加
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
def data_after_pca_output(file, datasets, features_header, output_path_after_pca, output_path_mlr_coef, indicator_val, num_of_cluster):
    username_val = datasets['username'].values
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values
    total_features = len(features_header)

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)


    pca = PCA(n_components=total_features)
    features_val = pca.fit_transform(features_val)

    pca_header_with_indicator = []
    for i in range(0, total_features):
        pca_header_with_indicator.append('comp' + str(i + 1))
    if num_of_cluster == 2:
        pca_header_with_indicator.append('indicator')
    else:
        for i in range(1, num_of_cluster):
            pca_header_with_indicator.append('indicator' + str(i))

    merge_metrics = []
    for i in range(0, len(features_val)):
        tmp = []
        for col in features_val[i]:
            tmp.append(col)
        for col in indicator_val[i]:
            tmp.append(col)
        merge_metrics.append(tmp)
    features_val = pd.DataFrame(merge_metrics, columns=pca_header_with_indicator)
    features_val = features_val.values

    result_metrics = []
    for i in range(0, len(username_val)):
        tmp_list = []
        tmp_list.append(username_val[i])
        for col in features_val[i]:
            tmp_list.append(col)
        tmp_list.append(label_val[i])
        result_metrics.append(tmp_list)
    datasets_features = ['username']
    if num_of_cluster == 2:
        datasets_features.append('indicator')
    else:
        for i in range(1,num_of_cluster):
            datasets_features.append('indicator' + str(i))
    datasets_features.append('final_score')

    datasets_features[1:1] = features_header
    metrics_dataframe = pd.DataFrame(result_metrics, columns=datasets_features)
    metrics_dataframe.to_csv(output_path_after_pca, index=False)
    comp_headers = pca_header_with_indicator.append('const')

    MLR = linear_model.LinearRegression()
    MLR.fit(features_val, label_val)
    label_val_predict = MLR.predict(features_val)
    mse = np.mean((label_val_predict - label_val) ** 2)
    print mse
    print "MLR coef: "
    print MLR.coef_
    print "intercept_", MLR.intercept_
    coef_result = []
    tmp = []
    for coef in MLR.coef_:
        tmp.append(coef)
    tmp.append(MLR.intercept_)
    coef_result.append(tmp)
    print coef_result
    metrics_dataframe = pd.DataFrame(coef_result, columns=comp_headers)
    metrics_dataframe.to_csv(output_path_mlr_coef, index=False)


    final_result = []
    for i in range(0,len(username_val)):
        tmp = []
        tmp.append(username_val[i])
        for col in features_val[i]:
            tmp.append(col)
        tmp.append(label_val_predict[i])
        tmp.append(label_val[i])
        final_result.append(tmp)
    datasets_features = ['username']
    if num_of_cluster == 2:
        datasets_features.append('indicator')
    else:
        for i in range(1, num_of_cluster):
            datasets_features.append('indicator' + str(i))
    datasets_features.append('predict_score')
    datasets_features.append('real_score')
    datasets_features[1:1] = features_header
    metrics_dataframe = pd.DataFrame(final_result, columns=datasets_features)
    metrics_dataframe.to_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/blended_vs_online_vs_offline/all_result_a/residual_' + file + '.csv', index=False)

#計算pMSE
def modified_mse(output_path_acc, data_num):
    datasets = pd.read_csv(output_path_acc, sep=',')
    headers = ['evaluation_num', 'kfold_split_num', 'number_of_comp', 'pMAPC', 'pMSE']
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

