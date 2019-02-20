# coding: utf-8
#程式目的:高三增能B班資料集先使用resampling平衡資料集，然後使用7種分類演算法建立模型並加入投票機制

import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN

def read_file(algo):
    dataset = pd.read_csv('chkb_data/' + algo + '/ncu_data_week_1-6.csv', sep=',')
    feature_header = list(dataset)[1:-1]
    feature_num = len(feature_header)
    username = dataset['username'].values
    X = dataset[feature_header].values
    Y = dataset['final_score'].values
    return username, X, Y

def train_model(index, features_val_train, label_val_train, features_val_test, Y_predict):
    if index == 0:
        clf = DecisionTreeClassifier(random_state=0)
    elif index == 1:
        clf = GaussianNB()
    elif index == 2:
        clf = LinearSVC(random_state=0)
    elif index == 3:
        clf = linear_model.LogisticRegression(random_state=0)
    elif index == 4:
        clf = MLPClassifier(random_state=0)
    elif index == 5:
        clf = RandomForestClassifier(random_state=0)
    elif index == 6:
        clf = SVC(random_state=0, kernel='linear')

    clf.fit(features_val_train, label_val_train)
    label_val_predict = clf.predict(features_val_test)
    for val in label_val_predict:
        Y_predict[i].append(val)
    return Y_predict

def oversampling(index, X, Y):
    if index == 0:
        ros = RandomOverSampler(random_state=0)
        try:
            X_resampled, Y_resampled = ros.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y
    elif index == 1:
        ada = ADASYN(random_state=0)
        try:
            X_resampled, Y_resampled = ada.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y
    elif index == 2:
        sm = SMOTE(random_state=0)
        try:
            X_resampled, Y_resampled = sm.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y
    elif index == 3:
        rus = RandomUnderSampler(random_state=0)
        try:
            X_resampled, Y_resampled = rus.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y
    elif index == 4:
        nm1 = NearMiss(random_state=0)
        try:
            X_resampled, Y_resampled = nm1.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y
    elif index == 5:
        enn = EditedNearestNeighbours(random_state=0)
        try:
            X_resampled, Y_resampled = enn.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y
    elif index == 6:
        renn = RepeatedEditedNearestNeighbours(random_state=0)
        try:
            X_resampled, Y_resampled = renn.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y
    elif index == 7:
        allknn = AllKNN(random_state=0)
        try:
            X_resampled, Y_resampled = allknn.fit_sample(X, Y)
        except Exception, e:
            print str(e)
            X_resampled, Y_resampled = X, Y

    return X_resampled, Y_resampled


algo_list = ['dt', 'GaNB', 'linear_svc', 'logistic', 'nn', 'rf', 'svc']

X_list = []
Y_list = []

for algo in algo_list:
    username_val, X, Y = read_file(algo)
    X_list.append(X)
    Y_list.append(Y)

number_of_folds = 10

number_of_cv_evaluation = 100
oversampler_name_list = ['ros', 'ada', 'sm', 'rus', 'nm1', 'enn', 'renn', 'allknn']
oversampler_name_list = ['allknn']
for oversampler_index in range(0,len(oversampler_name_list)):
    print 'oversampler:', oversampler_name_list[oversampler_index]
    metrics_list = []
    for evaluation_num in range(1, number_of_cv_evaluation + 1):
        print 'evaluation_num:', evaluation_num
        username_list = [[], [], [], [], [], [], []]
        Y_true = [[], [], [], [], [], [], []]
        Y_predict = [[], [], [], [], [], [], []]
        kfold = KFold(n_splits=number_of_folds, shuffle=True)
        for train_index, test_index in kfold.split(X_list[0]):
            for i in range(0,len(X_list)):
                username_test = username_val[test_index]
                X_train, X_test = X_list[i][train_index], X_list[i][test_index]
                Y_train, Y_test = Y_list[i][train_index], Y_list[i][test_index]
                
                X_train_res, Y_train_res = oversampling(oversampler_index, X_train, Y_train)

                standard_scaler = preprocessing.StandardScaler()
                standard_scaler.fit(X_train_res)
                features_val_train = standard_scaler.transform(X_train_res)
                features_val_test = standard_scaler.transform(X_test)

                print 'train_model:',i
                Y_predict = train_model(i, X_train_res, Y_train_res, X_test, Y_predict)
                for username in username_test:
                    username_list[i].append(username)
                for label_val in Y_test:
                    Y_true[i].append(label_val)

        metrics_dataframe = pd.DataFrame(username_list[0], columns=['username'])
        metrics_dataframe.insert(1, 'final_score', Y_true[0])

        for i in range(len(Y_predict)-1,-1,-1):
            metrics_dataframe.insert(1, algo_list[i], Y_predict[i])
        #metrics_dataframe.to_csv('chkb_result/test.csv', index=False)

        algo_header = list(metrics_dataframe)[1:]
        voting_result = metrics_dataframe[algo_header].values
        final_predict_result = []
        for i in range(0,len(voting_result)):
            result = Counter(voting_result[i]).most_common(1)
            final_predict_result.append(result[0][0])
        acc = accuracy_score(Y_true[0], final_predict_result)
        report = precision_recall_fscore_support(Y_true[0], final_predict_result, average='weighted')
        metrics_list.append([evaluation_num, acc, report[0], report[1], report[2]])
    metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'acc', 'precision', 'recall', 'f1_score'])
    metrics_dataframe.to_csv('chkb_result/voting_result_' + oversampler_name_list[oversampler_index] + '.csv', index=False)

