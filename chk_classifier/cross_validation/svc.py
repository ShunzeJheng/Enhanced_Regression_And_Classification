# coding: utf-8
#輸出SVC結果

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

#進行模型訓練與評估
filename_list = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '2-3', '2-4', '2-5', '2-6', '3-4', '3-5', '3-6', '4-5', '4-6', '5-6']
filename_list = ["1-6"]
for filename in filename_list:
    # 讀取資料並指定特徵與LABEL
    datasets = pd.read_csv('../all_data_a/ncu_data_week_' + filename + '.csv', sep=',')

    features_begin_index = 1
    features_end_index = 15

    features_header = list(datasets)[features_begin_index : features_end_index + 1]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values

    number_of_folds = 10
    total_features = features_end_index - features_begin_index + 1

    # 標準化
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val = standard_scaler.transform(features_val)

    number_of_cv_evaluation = 100

    metrics_list = []
    kfold_data = []
    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        kfold = KFold(n_splits=number_of_folds, shuffle=True)
        kfold_split_num = 1

        for train_index, test_index in kfold.split(features_val):
            features_val_train, features_val_test = features_val[train_index], features_val[test_index]
            label_val_train, label_val_test = label_val[train_index], label_val[test_index]

            # classifier
            clf = SVC(random_state=0, kernel='linear')
            clf.fit(features_val_train, label_val_train)
            label_val_predict = clf.predict(features_val_test)

            error_num = 0.0
            for i in range(0,len(label_val_test)):
                if label_val_test[i] != label_val_predict[i]:
                    error_num = error_num + 1.0
            acc = (len(label_val_test) - error_num) / len(label_val_test)

            metrics_list.append([evaluation_num, kfold_split_num, acc])
            for i in range(0,len(label_val_test)):
                kfold_data.append([evaluation_num, kfold_split_num, label_val_test[i], label_val_predict[i]])
            kfold_split_num = kfold_split_num + 1


    metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'acc'])
    metrics_dataframe = metrics_dataframe.groupby(['evaluation_num'], as_index=False).mean()
    #metrics_dataframe = metrics_dataframe.drop('evaluation_num', 1)
    #metrics_dataframe.to_csv('result_a/svc/dt_' + filename + '.csv', index=False)

    # 計算各項指標
    kfold_data = pd.DataFrame(kfold_data, columns=['evaluation_num', 'kfold_split_num', 'label_val_test', 'label_val_predict'])
    result = []
    tmp = 1
    for index in range(1, 101):
        data = kfold_data[kfold_data.evaluation_num == index]
        label_val_test = data['label_val_test'].values
        label_val_predict = data['label_val_predict'].values
        # print len(label_val_test)
        acc = accuracy_score(label_val_test, label_val_predict)
        report = precision_recall_fscore_support(label_val_test, label_val_predict, average='weighted')
        # print report
        result.append([tmp, index, acc, report[0], report[1], report[2]])
    kfold_data = pd.DataFrame(result, columns=['tmp', 'evaluation_num', 'acc', 'precision', 'recall', 'f1_score'])
    kfold_data = kfold_data.groupby(['tmp'], as_index=False).mean()
    kfold_data = kfold_data.drop('tmp', 1)
    kfold_data = kfold_data.drop('evaluation_num', 1)
    kfold_data.to_csv('result_a/svc/kfold_data_dt_' + filename + '.csv', index=False)