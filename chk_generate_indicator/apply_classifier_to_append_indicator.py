# coding: utf-8
#程式目的:輸出高三增能資料集在不同模型下對indicator的預測結果，並將結果append到資料集上

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

filename_list = ["1-5"]


for filename in filename_list:
    datasets = pd.read_csv('all_data_b/ncu_data_week_' + filename + '.csv', sep=',')

    goal_cluster_dataset = pd.read_csv('all_data_b/goal_cluster_remove_outlier_' + filename + '.csv', sep=',')
    goal_cluster = goal_cluster_dataset['indicator'].values

    # 每個資料集的特徵數不一樣，記得修改特徵數
    features_begin_index = 1
    features_end_index = 53
    features_mid_index = 53

    features_header = list(datasets)[features_begin_index : features_mid_index + 1]
    features_val = datasets[features_header].values
    label_header = 'final_score'
    label_val = datasets[label_header].values

    dataset_headers = list(datasets)[0 : features_end_index + 1]
    dataset_val = datasets[dataset_headers].values
    result_metrics = []
    standard_scaler = preprocessing.StandardScaler()
    standard_scaler.fit(features_val)
    features_val_standard = standard_scaler.transform(features_val)

    ##################################################################################################
    # 記得修改要使用的演算法

    #clf = KNeighborsClassifier(n_neighbors=5).fit(features_val_standard, goal_cluster)
    #clf = LinearSVC(random_state=0, loss='squared_hinge').fit(features_val_standard, goal_cluster)
    clf = SVC(random_state=0, kernel='linear').fit(features_val_standard, goal_cluster)
    #clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=53,
     #                            min_samples_leaf=3).fit(features_val_standard, goal_cluster)
    #clf = RandomForestClassifier(random_state=0, criterion='entropy', min_samples_leaf=3).fit(
     #   features_val_standard, goal_cluster)
    #clf = GaussianNB().fit(features_val_standard, goal_cluster)
    #clf = linear_model.LogisticRegression().fit(features_val_standard, goal_cluster)
    label_predict = clf.predict(features_val_standard)
    #joblib.dump(clf, 'all_result_b/svc.pkl')

    for index in range(0,len(features_val)):
        tmp = []
        for col in dataset_val[index]:
            tmp.append(col)
        tmp.append(label_predict[index])
        tmp.append(label_val[index])
        result_metrics.append(tmp)

    indictor_header = ['indicator','final_score']
    indictor_header[0:0] = dataset_headers
    metrics_dataframe = pd.DataFrame(result_metrics, columns=indictor_header)
    metrics_dataframe.to_csv('all_result_b/ncu_data_week_' + filename + '.csv', index=False)

    print "predict: "
    print label_predict
    print "goal: "
    print goal_cluster
    tn, fp, fn, tp = confusion_matrix(goal_cluster, label_predict).ravel()
    print tn, fp, fn, tp
    print 'acc: ' , float(tp+tn)/(len(label_val))