# coding: utf-8

import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def error_range(file):
    result = []

    datasets = pd.read_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/blended_vs_online_vs_offline/all_result_a/cross_validation_Residual_table_blended_' + file + '.csv', sep=',')

    features_header = list(datasets)[0:11]

    number_of_cv_evaluation = 100


    def posi_or_nega(error):
        sym_tmp = 0
        if error == 0:
            sym_tmp = 0
        elif error > 0:
            sym_tmp = 1
        else:
            sym_tmp = 2

        return sym_tmp


    for evaluation_num in range(1,number_of_cv_evaluation + 1):

        for score_range in range(10,110,10):
            features_val = datasets[datasets.evaluation_num == evaluation_num]
            if score_range == 100:
                features_val = features_val[features_val.label_val_test <= score_range]
                features_val = features_val[features_val.label_val_test >= score_range - 10]
            else:
                features_val = features_val[features_val.label_val_test < score_range]
                features_val = features_val[features_val.label_val_test >= score_range - 10]
            features_val = features_val[features_header].values
            posi_error_list, nega_error_list = [], []
            num_of_same, num_of_positive, num_of_negative = 0,0,0
            posi_error_rate_list = []
            nega_error_rate_list = []
            max_posi_error_rate = 0.0
            min_naga_error_rate = 0.0
            for data in features_val:
                # symbol = 0 -> same ； symbol = 1-> positive ；symbol = 2 -> negative
                error = data[2] - data[3]
                error_rate = error/data[3]
                symbol = posi_or_nega(error)
                if symbol == 1:
                    posi_error_list.append(error)
                    num_of_positive += 1
                    posi_error_rate_list.append(error_rate)
                elif symbol == 2:
                    nega_error_list.append(error)
                    num_of_negative += 1
                    nega_error_rate_list.append(error_rate)
                else:
                    num_of_same += 1

            score_range_string = str(score_range - 10) + ' to ' + str(score_range)
            mean_posi_error = np.mean(posi_error_list)
            mean_nega_error = np.mean(nega_error_list)
            if len(posi_error_rate_list) == 0:
                mean_posi_error_rate = 0
                max_posi_error_rate = 0
            else:
                mean_posi_error_rate = np.mean(posi_error_rate_list)
                max_posi_error_rate = max(posi_error_rate_list)

            if len(nega_error_rate_list) == 0:
                mean_nega_error_rate = 0
                min_naga_error_rate = 0
            else:
                mean_nega_error_rate = np.mean(nega_error_rate_list)
                min_naga_error_rate = min(nega_error_rate_list)

            if len(posi_error_list) == 0:
                max_posi_error = 0
            else:
                max_posi_error = max(posi_error_list)

            if len(nega_error_list) == 0:
                min_nega_error = 0
            else:
                min_nega_error = min(nega_error_list)


            result.append([evaluation_num, score_range_string, num_of_same, num_of_positive, num_of_negative , mean_posi_error, mean_nega_error, max_posi_error, min_nega_error, mean_posi_error_rate, mean_nega_error_rate, max_posi_error_rate, min_naga_error_rate])

    result = pd.DataFrame(result, columns=['evaluation_num', 'score', 'num_of_same', 'num_of_positive', 'num_of_negative', 'posi_error', 'nega_error', 'max_posi_error', 'min_nega_error', 'mean_posi_error_rate', 'mean_nega_error_rate', 'max_posi_error_rate', 'min_naga_error_rate'])
    result.to_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/blended_vs_online_vs_offline/all_result_a/error_table_' + file + '.csv', index=False)


    result = []
    features_val = datasets[features_header].values
    for data in features_val:
        positive_error = 0
        positive_error_rate = 0
        negative_error = 0
        negative_error_rate = 0
        error = data[2] - data[3]
        if error >= 0:
            positive_error = error
            positive_error_rate = positive_error / data[3]
        else:
            negative_error = error
            negative_error_rate = negative_error / data[3]

        result.append([data[0], data[1], data[2], data[3], positive_error, negative_error, positive_error_rate, negative_error_rate])

    result = pd.DataFrame(result, columns=['evaluation_num', 'kfold_split_num', 'label_val_predict', 'label_val_test', 'positive_error', 'negative_error', 'positive_error_rate', 'negative_error_rate'])
    result.to_csv('C:/Users/kslab/PycharmProjects/chk_with_indicator/blended_vs_online_vs_offline/all_result_a/error_table_2_' + file + '.csv', index=False)