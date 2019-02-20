# coding: utf-8
#程式目的:產生每位學生的username對應18週的資料表

import pandas as pd

edx_feature_by_week = pd.read_csv('data/student_features_week_raw_data_with_dropout_label_MO0015.csv', sep=',')

edx_feature_by_week = edx_feature_by_week.groupby(['username'], as_index=False).sum()
edx_feature_by_week = edx_feature_by_week['username']


l = []
for item in edx_feature_by_week.values:
	for i in range(1,19):
		l.append([item, i])
l = pd.DataFrame(l , columns=['username', 'week'])
l.to_csv('data/edx_feature_by_week_for_fill_empty_week.csv', index=False)
