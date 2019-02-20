# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing

filename_list = ["ncu_data_week_1-6(1a)", "ncu_data_week_1-18(3a)"]

for filename in filename_list:
    datasets = pd.read_csv('../data/' + filename + '.csv', sep=',')
    datasets_headers = list(datasets)[0:23]
    datasets = datasets[datasets.username != 102602012]
    datasets = datasets[datasets.username != 104602014]
    username_val = datasets['username'].values
    features_header = list(datasets)[1 : 22]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values

    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)

    result_metrics = []

    for i in range(0, len(username_val)):
        tmp_list = []
        tmp_list.append(username_val[i])
        for col in features_val[i]:
            tmp_list.append(col)
        tmp_list.append(label_val[i])
        result_metrics.append(tmp_list)

    metrics_dataframe = pd.DataFrame(result_metrics, columns=datasets_headers)
    metrics_dataframe.to_csv('result/' + filename + '_normalize.csv', index=False)
