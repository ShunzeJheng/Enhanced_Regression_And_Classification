# coding: utf-8
#程式目的:輸出地科班資料集每個特徵之pearson相關係數與p-value

import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd
from scipy.spatial.distance import correlation

filename_list = ["1-6(1a)", "1-12(2a)", "1-18(3a)", "7-12(2d)", "13-18(3d)"]
for filename in filename_list:
    dataset = pd.read_csv('ets_data/ncu_data_week_' + filename + '.csv', sep=',')
    features_header = list(dataset)[1: 56]
    label_header = 'final_score'
    label_val = dataset[label_header].values
    metrics = []
    result_header = ['feature', 'pearson', 'p-value', 'distance_correlation']
    for feature in features_header:
        features_val = dataset[feature].values
        result = pearsonr(features_val,label_val)
        result_2 = correlation(features_val, label_val)
        metrics.append([feature, result[0], result[1], result_2])
    metrics = pd.DataFrame(metrics, columns=result_header)
    metrics.to_csv('ets_result/correlation_' + filename + '.csv', index=False)

