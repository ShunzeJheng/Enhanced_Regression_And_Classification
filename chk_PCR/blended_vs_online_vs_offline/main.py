# coding: utf-8
#輸出使用關鍵因子作為特徵的資料集執行PCR後的結果
#在這邊只需要執行blended就好
#在產生a班或b班的資料前，記得改資料夾或資料集的名稱，如class_a_csv或class_b_csv

import pandas as pd
from func import *
from coef_ import *
from coef_mlr import *


def blended_vs_online_vs_offline(file, key_feature):
    # 讀取資料並指定關鍵因子與LABEL
    datasets_small_name = file
    datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_PCR/all_data_a/ncu_data_week_' + datasets_small_name + '.csv', sep=',')

    user_header = 'username'
    user_val = datasets[user_header].values

    label_header = 'final_score'
    label_val = datasets[label_header].values

    data_num = len(user_val)
    features_header = key_feature
    features_val = datasets[features_header].values
    total_features = len(features_header)

    #指定輸出路徑
    output_path_acc = 'C:/Users/kslab/PycharmProjects/chk_PCR/blended_vs_online_vs_offline/all_result_a/PCR_blended_' + datasets_small_name + '.csv'
    output_path_residual = 'C:/Users/kslab/PycharmProjects/chk_PCR/blended_vs_online_vs_offline/all_result_a/cross_validation_Residual_table_blended_' + datasets_small_name + '.csv'
    output_path_after_pca = 'C:/Users/kslab/PycharmProjects/chk_PCR/blended_vs_online_vs_offline/all_result_a/data_after_pca_blended_' + datasets_small_name + '.csv'
    output_path_mlr_coef = 'C:/Users/kslab/PycharmProjects/chk_PCR/blended_vs_online_vs_offline/all_result_a/MLR_coef_blended_' + datasets_small_name + '.csv'
    output_path_data_with_predict = 'C:/Users/kslab/PycharmProjects/chk_PCR/blended_vs_online_vs_offline/all_result_a/data_with_predict_blended_' + datasets_small_name + '.csv'
    output_path_data_standard = 'C:/Users/kslab/PycharmProjects/chk_PCR/blended_vs_online_vs_offline/all_result_a/data_with_standard_' + datasets_small_name + '.csv'

    # 執行PCR，並輸出每一次交叉驗證的結果，包括預測分數、實際分數、預測準確度
    PCR(features_val, label_val, total_features, user_val,output_path_acc, output_path_residual, output_path_data_with_predict, features_header)
    #計算pMSE
    modified_mse(output_path_acc, data_num)
    #輸出特徵係數的p-value與加入r square、adjusted r square
    OLS_Regression(output_path_acc, datasets_small_name, total_features, features_val, label_val, 'blended')
    #輸出經過pca轉換後的資料集
    data_after_pca_output(datasets_small_name, datasets, features_header, output_path_after_pca, output_path_mlr_coef)
    #輸出標準化過後的資料集
    data_standard(user_val, features_val, label_val, output_path_data_standard, features_header)


    """
    #features_header = ['active_sum_count', 'video_sum_count', 'video_backward_seek_sum', 'mt_mean']
    #features_header = ['mt_mean']
    features_header = ['mt_unit_sum', 'mt_online_practice_num_day']
    features_val = datasets[features_header].values
    total_features = len(features_header)
    output_path_acc = 'all_result_a/PCR_online.csv'
    output_path_residual = 'all_result_a/cross_validation_Residual_table_online.csv'
    output_path_after_pca = 'all_result_a/data_after_pca_online.csv'
    output_path_mlr_coef = 'all_result_a/MLR_coef_online.csv'
    output_path_data_with_predict = 'all_result_a/data_with_predict_online.csv'
    PCR(features_val, label_val, total_features, user_val,output_path_acc, output_path_residual, output_path_data_with_predict, features_header)
    OLS_Regression(total_features, features_val, label_val, 'online')
    modified_mse(output_path_acc, data_num)
    
    
    features_header = ['hw_mean', 'qz_mean', 'counseling_join_num']
    features_header = ['hw_mean', 'qz_mean', 'counseling_join_num']
    #features_header = ['hw_mean', 'qz_mean', 'counseling_join_num']
    features_val = datasets[features_header].values
    total_features = len(features_header)
    output_path_acc = 'all_result_a/PCR_offline.csv'
    output_path_residual = 'all_result_a/cross_validation_Residual_table_offline.csv'
    output_path_after_pca = 'all_result_a/data_after_pca_offline.csv'
    output_path_mlr_coef = 'all_result_a/MLR_coef_offline.csv'
    output_path_data_with_predict = 'all_result_a/data_with_predict_offline.csv'
    PCR(features_val, label_val, total_features, user_val,output_path_acc, output_path_residual, output_path_data_with_predict, features_header)
    OLS_Regression(total_features, features_val, label_val, 'offline')
    modified_mse(output_path_acc, data_num)
    
    """


