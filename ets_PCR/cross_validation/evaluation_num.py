# coding: utf-8

import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


datasets = pd.read_csv('../data/ncu_p3a(week_1-18).csv', sep=',')

features_begin_index = 1
features_end_index = 51

features_header = list(datasets)[features_begin_index : features_end_index + 1]
features_val = datasets[features_header].values
label_header = 'final_score'
label_val = datasets[label_header].values

number_of_folds = 10

final_metrics_list = []


for number_of_cv_evaluation in range(1, 301):

	
	metrics_list = []

	for evaluation_num in range(number_of_cv_evaluation):
		kfold = KFold(n_splits=number_of_folds, shuffle=True)
		kfold_split_num = 1
		for train_index, test_index in kfold.split(features_val):
			features_val_train, features_val_test = features_val[train_index], features_val[test_index]
			label_val_train, label_val_test = label_val[train_index], label_val[test_index]
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
			R_squared = MLR.score(features_val_test, label_val_test)
			metrics_list.append([evaluation_num, kfold_split_num, MAPC, MSE, R_squared])
			kfold_split_num = kfold_split_num + 1

	metrics_dataframe = pd.DataFrame(metrics_list, columns=['evaluation_num', 'kfold_split_num', 'MAPC', 'MSE', 'R_squared'])
	metrics_dataframe = metrics_dataframe.mean()


	final_metrics_list.append([number_of_cv_evaluation, metrics_dataframe['MSE']])
	#print("Mean Evaluation MSE:" + str(metrics_dataframe['MSE']))
	#print("Mean Evaluation MAPC:" + str(metrics_dataframe['MAPC']))
	#print("Mean Evaluation R_squared:" + str(metrics_dataframe['R_squared']))
final_metrics_dataframe = pd.DataFrame(final_metrics_list, columns=['number_of_cv_evaluation', 'MSE'])
final_metrics_dataframe.to_csv('result/number_of_cv_evaluation.csv', index=False)
plt.plot(final_metrics_dataframe['number_of_cv_evaluation'], final_metrics_dataframe['MSE'], linewidth=1.0)
plt.xlabel('number of cross-validation evaluation')
plt.ylabel('Predictive Mean squared error (pMSE)')
plt.savefig('result/number_of_cv_evaluation.png')