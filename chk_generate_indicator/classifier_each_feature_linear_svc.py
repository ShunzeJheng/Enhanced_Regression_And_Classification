# coding: utf-8
#程式目的:輸出高三增能資料集在不同參數下的Linear_svc模型對indicator的準確度

from sklearn.svm import LinearSVC
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix


filename_list = ['1-1', '1-2', '1-3', '1-4', '1-5']
del_flag = True

for filename in filename_list:
    datasets = pd.read_csv('all_data_a/ncu_data_week_' + filename + '.csv', sep=',')
    label_header = 'final_score'
    label_val = datasets[label_header].values
    features_num = len(list(datasets)) - 2

    goal_cluster_dataset = pd.read_csv('all_data_a/goal_cluster_remove_outlier_' + filename + '.csv', sep=',')
    goal_cluster = goal_cluster_dataset['indicator'].values

    loss_types = ['hinge', 'squared_hinge']
    for loss_type in loss_types:
        result_metrics = []
        for features_begin_index in range(1,features_num+1):
            for features_mid_index in range(features_begin_index,features_num+1):
                features_header = list(datasets)[features_begin_index: features_mid_index + 1]
                features_val = datasets[features_header].values

                standard_scaler = preprocessing.StandardScaler()
                standard_scaler.fit(features_val)
                features_val_standard = standard_scaler.transform(features_val)

                clf = LinearSVC(random_state=0,loss=loss_type).fit(features_val_standard, goal_cluster)
                label_predict = clf.predict(features_val_standard)

                print "predict: "
                print label_predict
                print "goal: "
                print goal_cluster
                tn, fp, fn, tp = confusion_matrix(goal_cluster, label_predict).ravel()
                acc = float(tp + tn) / (len(label_val))
                print tn, fp, fn, tp
                print 'acc: ', acc
                result_metrics.append([features_begin_index, features_mid_index, tn, fp, fn, tp, acc])
        metrics_dataframe = pd.DataFrame(result_metrics, columns=['features_begin_index', 'features_mid_index', 'tn', 'fp', 'fn', 'tp', 'acc'])
        if del_flag:
            metrics_dataframe.to_csv('classifier_result_a/linear_svc_result/linear_svc_result_' + loss_type + '_' + filename + '_delete.csv', index=False)
        else:
            metrics_dataframe.to_csv('classifier_result_a/linear_svc_result/linear_svc_result_' + loss_type + '_' + filename + '.csv', index=False)