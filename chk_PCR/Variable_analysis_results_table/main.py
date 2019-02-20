# coding: utf-8
#輸出各個component的組成係數
#在產生a班或b班的資料前，記得改資料夾或資料集的名稱，如class_a_csv或class_b_csv

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn import linear_model
from scipy.stats import boxcox


def transformation(final_score):
    return final_score, 0
    # 做BOXCOX轉換，之後沒有用到
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

def Variable_analysis_results_table(file, index_min, features_begin_index, features_end_index):
    # 讀取資料並指定特徵與LABEL
    datasets_small_name = file
    datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_PCR/all_data_a/ncu_data_week_' + datasets_small_name + '.csv', sep=',')

    total_features = features_end_index - features_begin_index + 1

    features_header = list(datasets)[features_begin_index : features_end_index + 1]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values
    label_val, transfor_lamda = transformation(label_val)
    total_features = len(features_header)

    # 標準化
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)

    number_of_comp = index_min

    pca = PCA(n_components=number_of_comp)
    pca.fit(features_val)
    features_val = pca.transform(features_val)


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


    features_val = sm.add_constant(features_val) #sklearn 預設有加入截距，statsmodels沒有，所以要加
    results = sm.OLS(label_val, features_val).fit()
    print results.summary()


    pvalue_l = ['Regression P>|t|']
    for i in range(1, len(results.pvalues)):
        pvalue_l.append(round(results.pvalues[i], 3))

    variable_analysis_results_table.append(pvalue_l)
    variable_analysis_results_table = pd.DataFrame(variable_analysis_results_table, columns=columns)
    variable_analysis_results_table.to_csv('C:/Users/kslab/PycharmProjects/chk_PCR/Variable_analysis_results_table/all_result_a/' + datasets_small_name + '_best_comp_variable_analysis_results_table.csv', index=False)



