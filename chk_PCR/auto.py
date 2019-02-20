# coding: utf-8
#程式目的:自動化PCR流程
#在產生a班或b班的資料前，記得改資料夾或資料集的名稱，如class_a_csv或class_b_csv

from cross_validation.PCR import cross_validation
from Variable_analysis_results_table.main import Variable_analysis_results_table
from blended_vs_online_vs_offline.main import blended_vs_online_vs_offline
from blended_vs_online_vs_offline.error_range import error_range
import pandas as pd
import numpy as np


#指定要執行PCR的資料集
filename_list = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '2-3', '2-4', '2-5', '2-6', '3-4', '3-5', '3-6', '4-5', '4-6', '5-6']
filename_list = ['1-6']
#開始進行PCR
for file in filename_list:
    #告訴程式資料集中哪些欄位是特徵
    features_begin_index = 1
    features_end_index = 55
    #輸出資料集在各個component的表現
    print file,' cross_validation'
    cross_validation(file, features_begin_index, features_end_index)

    #找出資料集在哪個component表現最佳，並執行Variable_analysis_results_table，輸出各個component的組成係數
    datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_PCR/cross_validation/all_result_a/PCR_metrics_list_' + file + '_normalize.csv', sep=',')
    number_of_comp = datasets['number_of_comp'].values
    pMSE = datasets['pMSE'].values
    index_min = int(number_of_comp[np.argmin(pMSE)])
    print file, ' Variable_analysis_results_table'
    Variable_analysis_results_table(file, index_min, features_begin_index, features_end_index)

    #根據各個component的組成係數選擇關鍵因子
    datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_PCR/all_data_a/ncu_data_week_' + file + '.csv', sep=',')
    features_header = list(datasets)[features_begin_index: features_end_index + 1]
    datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_PCR/Variable_analysis_results_table/all_result_a/' + file + '_best_comp_variable_analysis_results_table.csv', sep=',')
    key_feature = []
    comp_header = []
    for i in range(1,index_min+1):
        comp_header.append('comp ' + str(i))

    #在每個有顯著的component中，選擇組成係數在0.5以上的特徵作為關鍵因子
    for comp in comp_header:
        comp_val = datasets[comp].values
        print comp_val
        if comp_val[len(comp_val)-1] < 0.05:
            for index in range(0,len(comp_val)):
                if comp_val[index] >= 0.5 or comp_val[index] <= -0.5:
                    if features_header[index] not in key_feature:
                        key_feature.append(features_header[index])

    #若組成係數皆不在0.5以上，則選擇0.4以上
    if not key_feature:
        for comp in comp_header:
            comp_val = datasets[comp].values
            print comp_val
            if comp_val[len(comp_val) - 1] < 0.05:
                for index in range(0, len(comp_val)):
                    if comp_val[index] >= 0.3 or comp_val[index] <= -0.4:
                        if features_header[index] not in key_feature:
                            key_feature.append(features_header[index])

    # 若組成係數皆不在0.4以上，則選擇0.3以上
    if not key_feature:
        for comp in comp_header:
            comp_val = datasets[comp].values
            print comp_val
            if comp_val[len(comp_val) - 1] < 0.05:
                for index in range(0, len(comp_val)):
                    if comp_val[index] >= 0.3 or comp_val[index] <= -0.3:
                        if features_header[index] not in key_feature:
                            key_feature.append(features_header[index])

    #視情況把hw_mean跟qz_mean加入關鍵因子
    """if 'hw_mean' not in key_feature:
        key_feature.append('hw_mean')
    if 'qz_mean' not in key_feature:
        key_feature.append('qz_mean')"""

    print 'key_feature:', key_feature
    #輸出使用關鍵因子作為特徵的資料集執行PCR後的結果
    if key_feature:
        print file, ' blended_vs_online_vs_offline'
        blended_vs_online_vs_offline(file, key_feature)