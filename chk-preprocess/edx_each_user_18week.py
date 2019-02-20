# coding: utf-8
#程式目的:產生每位學生的username對應18週的資料表

#在產生a班或b班的資料前，記得改資料夾或資料集的名稱，如class_a_csv或class_b_csv
import pandas as pd

final_score = pd.read_csv('class_a_csv/final_score.csv', sep=',')
username = final_score['username']

l = []
for item in username.values:
	for i in range(1,7):
		l.append([item, i])
l = pd.DataFrame(l , columns=['username', 'week'])
l.to_csv('class_a_csv/edx_feature_by_week_for_fill_empty_week.csv', index=False)
