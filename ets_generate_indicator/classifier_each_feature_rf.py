# coding: utf-8
#程式目的:輸出地科班資料集在不同參數下的RF模型對indicator的準確度

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix


filename_list = ["1-6(1a)", "1-12(2a)", "7-12(2d)", "13-18(3d)"]
del_flag = True

for filename in filename_list:
    datasets = pd.read_csv('data/ncu_data_week_' + filename + '.csv', sep=',')
    label_header = 'final_score'
    label_val = datasets[label_header].values
    features_num = len(list(datasets)) - 2

    goal_cluster_dataset = pd.read_csv('data/goal_cluster_remove_outlier_' + filename + '.csv', sep=',')
    goal_cluster = goal_cluster_dataset['indicator'].values
    criterion_types = ['gini','entropy']
    for criterion_type in criterion_types:
        result_metrics = []
        for leaf_node_num in range(3,10):
            for features_begin_index in range(1,features_num+1):
                for features_mid_index in range(features_begin_index,features_num+1):
                    features_header = list(datasets)[features_begin_index: features_mid_index + 1]
                    features_val = datasets[features_header].values

                    standard_scaler = preprocessing.StandardScaler()
                    standard_scaler.fit(features_val)
                    features_val_standard = standard_scaler.transform(features_val)

                    clf = RandomForestClassifier(random_state=0,criterion=criterion_type, min_samples_leaf = leaf_node_num).fit(features_val_standard, goal_cluster)
                    label_predict = clf.predict(features_val_standard)

                    print "predict: "
                    print label_predict
                    print "goal: "
                    print goal_cluster
                    tn, fp, fn, tp = confusion_matrix(goal_cluster, label_predict).ravel()
                    acc = float(tp + tn) / (len(label_val))
                    print tn, fp, fn, tp
                    print 'acc: ', acc
                    result_metrics.append([features_begin_index, features_mid_index, leaf_node_num, tn, fp, fn, tp, acc])
        metrics_dataframe = pd.DataFrame(result_metrics, columns=['features_begin_index', 'features_mid_index', 'leaf_node_num', 'tn', 'fp', 'fn', 'tp', 'acc'])
        if del_flag:
            metrics_dataframe.to_csv('classifier_result/rf_result/rf_result_' + criterion_type + '_' + filename + '_delete.csv', index=False)
        else:
            metrics_dataframe.to_csv('classifier_result/rf_result/rf_result_' + criterion_type + '_' + filename + '.csv', index=False)