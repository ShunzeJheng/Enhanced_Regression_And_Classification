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
from sklearn.metrics import confusion_matrix
filename_list = ["1-18(3a)"]
for filename in filename_list:
    dataset = pd.read_csv('ets_data/ncu_data_week_' + filename + '.csv', sep=',')
    features_header = list(dataset)[1: 57]
    label_header = 'final_score'
    label_val = dataset[label_header].values
    label_list = list(set(label_val))
    print label_list
    metrics = []
    metrics_header = ['feature', 'logistic', 'linear_svc', 'svc', 'GaNB', 'Decision_tree', 'Random_forest', 'NN']
    features_val = dataset['qz_mean'].values
    features_val = np.reshape(features_val, (59,1))
    print np.shape(features_val)
    clf = linear_model.LogisticRegression()
    clf.fit(features_val, label_val)
    label_val_predict = clf.predict(features_val)

    auc_list = []
    for label in label_list:
        print label
        y_true = []
        y_predit = []
        print 'label_val',label_val
        print 'label_val_predict',label_val_predict
        for i in range(0,len(label_val)):
            if label_val[i] == label:
                y_true.append(0)
            else:
                y_true.append(1)
            if label_val_predict[i] == label:
                y_predit.append(0)
            else:
                y_predit.append(1)
        print 'y_true',y_true
        print 'y_predit',y_predit
        auc = roc_auc_score(y_true, y_predit)
        auc_list.append(auc)
    print auc_list

