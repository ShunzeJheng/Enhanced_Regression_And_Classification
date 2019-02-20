# coding: utf-8

import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


datasets = pd.read_csv('../data/ncu_data_week_1-6(1a).csv', sep=',')

features_begin_index = 1
features_end_index = 21

features_header = list(datasets)[features_begin_index : features_end_index + 1]
features_val = datasets[features_header].values
label_header = 'final_score'
label_val = datasets[label_header].values

number_of_folds = 10
number_of_cv_evaluation = 100


metrics_list = []

for evaluation_num in range(1, number_of_cv_evaluation + 1):
	kfold = KFold(n_splits=number_of_folds, shuffle=True)
	kfold_split_num = 1
	for train_index, test_index in kfold.split(features_val):
		features_val_train, features_val_test = features_val[train_index], features_val[test_index]
		label_val_train, label_val_test = label_val[train_index], label_val[test_index]
		
		
		min_max_scaler = preprocessing.MinMaxScaler()
		min_max_scaler.fit(features_val_train)
		
		#features_val_train = min_max_scaler.transform(features_val_train)
		#features_val_test = min_max_scaler.transform(features_val_test)


		MLR = linear_model.LinearRegression()
		MLR.fit(features_val_train, label_val_train)
		label_val_predict = MLR.predict(features_val_test)


		#處理預測值超過不合理範圍
		for i in range(len(label_val_predict)):
			if label_val_predict[i] > 100.0:
				label_val_predict[i] = 100.0
			elif label_val_predict[i] < 0:
				label_val_predict[i] = 0.0
		#處理預測值超過不合理範圍
		
		
		MAPC = 1 - np.mean(abs((label_val_predict - label_val_test) / label_val_test))
		MSE = np.mean((label_val_predict - label_val_test) ** 2)
		metrics_list.append([evaluation_num, kfold_split_num, MAPC, MSE])
		kfold_split_num = kfold_split_num + 1


metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'MAPC', 'MSE'])
#metrics_dataframe.to_csv('result/t.csv', index=False)
metrics_dataframe = metrics_dataframe.mean()


print("Mean pMSE:" + str(metrics_dataframe['MSE']))
print("Mean pMAPC:" + str(metrics_dataframe['MAPC']))
