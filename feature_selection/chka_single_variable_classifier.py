# coding: utf-8
#程式目的:輸出高三增能A班資料集使用single variable後的結果(每個特徵的準確度)

import numpy as np
from scipy.stats.stats import pearsonr
import pandas as pd
from scipy.spatial.distance import correlation
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

def multiclass_auc(label_list, label_val, label_val_predict):
    auc_list = []
    for label in label_list:
        #print label
        y_true = []
        y_predit = []
        #print 'label_val',label_val
        #print 'label_val_predict',label_val_predict
        for i in range(0,len(label_val)):
            if label_val[i] == label:
                y_true.append(0)
            else:
                y_true.append(1)
            if label_val_predict[i] == label:
                y_predit.append(0)
            else:
                y_predit.append(1)
        #print 'y_true',y_true
        #print 'y_predit',y_predit
        auc = roc_auc_score(y_true, y_predit)
        auc_list.append(auc)
    return np.mean(auc_list)

filename_list = ["1-6"]
for filename in filename_list:
    dataset = pd.read_csv('chka_data/ncu_data_week_' + filename + '.csv', sep=',')
    features_header = list(dataset)[2: 57]
    label_header = 'final_score'
    label_val = dataset[label_header].values
    label_list = list(set(label_val))
    metrics = []
    metrics_header = ['feature', 'logistic', 'linear_svc', 'svc', 'GaNB', 'Decision_tree', 'Random_forest', 'NN']
    for feature in features_header:
        features_val = dataset[feature].values
        features_val = np.reshape(features_val, (len(label_val), 1))
        clf_list = []
        clf = linear_model.LogisticRegression()
        clf_list.append(clf)
        clf = LinearSVC()
        clf_list.append(clf)
        clf = SVC(random_state=0, kernel='linear')
        clf_list.append(clf)
        clf = GaussianNB()
        clf_list.append(clf)
        clf = DecisionTreeClassifier()
        clf_list.append(clf)
        clf = RandomForestClassifier()
        clf_list.append(clf)
        clf = MLPClassifier()
        clf_list.append(clf)
        result = []
        result.append(feature)
        for clf in clf_list:
            clf.fit(features_val, label_val)
            label_val_predict = clf.predict(features_val)
            auc = multiclass_auc(label_list, label_val, label_val_predict)
            result.append(auc)
        metrics.append(result)
    metrics = pd.DataFrame(metrics, columns=metrics_header)
    metrics.to_csv('chka_result/classifier_' + filename + '.csv', index=False)

