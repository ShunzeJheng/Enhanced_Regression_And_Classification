# coding: utf-8
#程式目的:產生每位學生1-6週的特徵
#在產生a班或b班的資料前，記得改資料夾或資料集的名稱，如class_a_csv或class_b_csv

import pandas as pd

#讀取各個資料表
edx_feature_by_week = pd.read_csv('class_a_csv/student_features_week_raw_data_with_dropout_label_chka.csv', sep=',')
edx_feature_by_week_for_fill_empty_week = pd.read_csv('class_a_csv/edx_feature_by_week_for_fill_empty_week.csv', sep=',')
maple_TA_feature_by_week = pd.read_csv('maple_TA_a_feature_in_week/maple_TA_a_feature_in_week_sixth.csv', sep=',')
tmp = pd.read_csv('class_a_csv/tmp.csv', sep=',')

#各個資料表進行合併
features_df = edx_feature_by_week.merge(edx_feature_by_week_for_fill_empty_week, left_on=["username", "week"], right_on=["username", "week"], how='outer')
features_df = features_df.merge(maple_TA_feature_by_week, left_on=["username", "week"], right_on=["username", "week"], how='outer')
features_df = features_df.merge(tmp, left_on=["username"], right_on=["username"], how='outer')

#補值，以0取代-1與nan，並捨棄不需要的欄位
features_df = features_df.replace([-1], [0]) #-1 => 0
features_df = features_df.fillna(0) #NaN => 0
features_df = features_df.drop('email', 1)
features_df = features_df.drop('course_id', 1)
features_df = features_df.drop('first_three_week_homework_complete_num', 1)
features_df = features_df.drop('last_three_week_homework_complete_num', 1)
features_df = features_df.drop('repeat_num', 1)

features_df.to_csv('result_a/all_features_by_week.csv', index=False)
