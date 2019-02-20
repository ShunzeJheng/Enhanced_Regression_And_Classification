import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt


datasets = pd.read_csv('../mean_data_b/ncu_data_week_1-6.csv', sep=',')

features_begin_index = 2
features_end_index = 23

features_header = list(datasets)[features_begin_index : features_end_index + 1]
features_val = datasets[features_header].values
label_header = 'final_score'
label_val = datasets[label_header].values
final_score_mean_a = 4504.00/119
final_score_mean_b = 5020.00/162


kfold = KFold(n_splits=10, shuffle=True)
kfold_split_num = 1

print features_val
"""
for train_index, test_index in kfold.split(features_val):
	features_val_train, features_val_test = features_val[train_index], features_val[test_index]
	label_val_train, label_val_test = label_val[train_index], label_val[test_index]
	MLR = linear_model.LinearRegression()
	MLR.fit(features_val_train, label_val_train)
	label_val_predict = MLR.predict(features_val_test)
print label_val_predict
"""