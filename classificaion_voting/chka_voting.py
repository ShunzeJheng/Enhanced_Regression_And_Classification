# coding: utf-8
#程式目的:高三增能A班資料集使用7種分類演算法建立模型並加入投票機制

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

def read_file(algo):
    dataset = pd.read_csv('chka_data/' + algo + '/ncu_data_week_1-6.csv', sep=',')
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

algo_list = ['dt', 'GaNB', 'linear_svc', 'logistic', 'nn', 'rf', 'svc']

X_list = []
Y_list = []

for algo in algo_list:
    username_val, X, Y = read_file(algo)
    X_list.append(X)
    Y_list.append(Y)

number_of_folds = 10

metrics_list = []
number_of_cv_evaluation = 100
for evaluation_num in range(1, number_of_cv_evaluation + 1):
    print 'evaluation_num:', evaluation_num
    username_list = [[], [], [], [], [], [], []]
    Y_true = [[], [], [], [], [], [], []]
    Y_predict = [[], [], [], [], [], [], []]
    kfold = KFold(n_splits=number_of_folds, shuffle=True)
    for train_index, test_index in kfold.split(X_list[0]):
        for i in range(0,len(X_list)):
            username_test = username_val[test_index]
            features_val_train, features_val_test = X_list[i][train_index], X_list[i][test_index]
            label_val_train, label_val_test = Y_list[i][train_index], Y_list[i][test_index]

            standard_scaler = preprocessing.StandardScaler()
            standard_scaler.fit(features_val_train)
            features_val_train = standard_scaler.transform(features_val_train)
            features_val_test = standard_scaler.transform(features_val_test)

            Y_predict = train_model(i, features_val_train, label_val_train, features_val_test, Y_predict)
            for username in username_test:
                username_list[i].append(username)
            for label_val in label_val_test:
                Y_true[i].append(label_val)

    metrics_dataframe = pd.DataFrame(username_list[0], columns=['username'])
    metrics_dataframe.insert(1, 'final_score', Y_true[0])

    for i in range(len(Y_predict)-1,-1,-1):
        metrics_dataframe.insert(1, algo_list[i], Y_predict[i])
    metrics_dataframe.to_csv('chka_result/test.csv', index=False)

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
metrics_dataframe.to_csv('chka_result/voting_result.csv', index=False)

