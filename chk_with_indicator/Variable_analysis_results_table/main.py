# coding: utf-8
#輸出各個component的組成係數

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn import linear_model

def Variable_analysis_results_table(file, index_min, features_begin_index, features_end_index, num_of_cluster):
    # 讀取資料並指定特徵與LABEL
    datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/all_data_a/ncu_data_week_' + file + '.csv', sep=',')

    total_features = features_end_index - features_begin_index + 1

    features_header = list(datasets)[features_begin_index : features_end_index + 1]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values
    total_features = len(features_header)

    # 判斷有幾個indicator，並加入header
    indicator_header = []
    if num_of_cluster == 2:
        indicator_header.append('indicator')
    else:
        for i in range(1, num_of_cluster):
            indicator_header.append('indicator' + str(i))
    indicator_val = datasets[indicator_header].values

    # 標準化
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)

    number_of_comp = index_min

    pca = PCA(n_components=number_of_comp)
    pca.fit(features_val)
    features_val = pca.transform(features_val)

    pca_header_with_indicator = []
    for i in range(0,number_of_comp):
        pca_header_with_indicator.append('comp' + str(i+1))
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
        for val in indicator_val[i]:
            tmp.append(val)
        merge_metrics.append(tmp)

    features_val = pd.DataFrame(merge_metrics, columns=pca_header_with_indicator)
    features_val = features_val.values

    variable_analysis_results_table = []
    for i in range(total_features):
        row_list = []
        row_list.append(features_header[i])
        for j in pca.components_:
            row_list.append(j[i])
        variable_analysis_results_table.append(row_list)



    columns = ['']

    for i in range(1, number_of_comp + 1):
        columns.append("comp " + str(i))

    if num_of_cluster == 2:
        columns.append('indicator')
    else:
        for i in range(1, num_of_cluster):
            columns.append('indicator' + str(i))




    features_val = sm.add_constant(features_val) #sklearn 預設有加入截距，statsmodels沒有，所以要加
    results = sm.OLS(label_val, features_val).fit()
    print results.summary()



    pvalue_l = ['Regression P>|t|']
    for i in range(1, len(results.pvalues)):
        pvalue_l.append(round(results.pvalues[i], 3))

    variable_analysis_results_table.append(pvalue_l)
    print pvalue_l
    variable_analysis_results_table = pd.DataFrame(variable_analysis_results_table, columns=columns)
    variable_analysis_results_table.to_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/Variable_analysis_results_table/all_result_a/' + file + '_best_comp_variable_analysis_results_table.csv', index=False)

