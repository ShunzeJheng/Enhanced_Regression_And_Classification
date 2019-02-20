# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn import linear_model


datasets_small_name = '1a'
datasets = pd.read_csv('../all_data_a/ncu_data_week_2-5.csv', sep=',')




features_begin_index = 2
features_end_index = 23
#total_features = 22

#features_header = list(datasets)[features_begin_index : features_end_index + 1]
features_header = ['active_sum_count', 'video_num_days', 'stop_video_sum', 'video_forward_seek_sum', 'video_backward_seek_sum', 'video_pause_sum', 'Cumulative_num_days', 'hw_mean', 'exam_mean']
#features_header = ['active_num_days', 'seek_video_sum', 'video_forward_seek_sum', 'video_backward_seek_sum', 'video_pause_sum', 'Problem_num_day', 'hw_mean', 'exam_mean']
total_features = len(features_header)
features_val = datasets[features_header].values
label_header = 'final_score'
label_val = datasets[label_header].values

standard_scaler = preprocessing.StandardScaler()
standard_scaler.fit(features_val)
features_val = standard_scaler.transform(features_val)


number_of_comp = 9

pca = PCA(n_components=number_of_comp)
pca.fit(features_val)
features_val = pca.transform(features_val)


variable_analysis_results_table = []
for i in range(total_features):
	row_list = []
	row_list.append(features_header[i])
	for j in pca.components_:
		row_list.append(j[i])
	variable_analysis_results_table.append(row_list)



columns = ['']

for i in range(1, number_of_comp + 1):
	columns.append("comp " + str(i))





features_val = sm.add_constant(features_val) #sklearn 預設有加入截距，statsmodels沒有，所以要加
results = sm.OLS(label_val, features_val).fit()
print results.summary()



pvalue_l = ['Regression P>|t|']
for i in range(1, len(results.pvalues)):
	pvalue_l.append(round(results.pvalues[i], 3))

variable_analysis_results_table.append(pvalue_l)
variable_analysis_results_table = pd.DataFrame(variable_analysis_results_table, columns=columns)
variable_analysis_results_table.to_csv('all_result_a/' + '2-5_best_comp_variable_analysis_results_table_normalize.csv', index=False)



