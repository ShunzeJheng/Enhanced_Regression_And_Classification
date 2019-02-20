# coding: utf-8
#程式目的:產生每位學生1-18週的特徵

import pandas as pd

#讀取各個資料表
edx_feature_by_week = pd.read_csv('data/student_features_week_raw_data_with_dropout_label_MO0015.csv', sep=',')
edx_feature_by_week_for_fill_empty_week = pd.read_csv('data/edx_feature_by_week_for_fill_empty_week.csv', sep=',')
maple_TA_feature_by_week = pd.read_csv('data/maple_TA_feature_by_week.csv', sep=',')
homework_by_week = pd.read_csv('data/homework_by_week.csv', sep=',')
#mapleta_by_week.csv是紀錄mt_mean這個欄位的資料表，但後來因為高三增能沒有這個欄位，所以不使用，對應到line 20
#mapleta_by_week = pd.read_csv('data/mapleta_by_week.csv', sep=',')
quiz_by_week = pd.read_csv('data/quiz_by_week.csv', sep=',')


#各個資料表進行合併
features_df = edx_feature_by_week.merge(edx_feature_by_week_for_fill_empty_week, left_on=["username", "week"], right_on=["username", "week"], how='outer')
features_df = features_df.merge(maple_TA_feature_by_week, left_on=["username", "week"], right_on=["username", "week"], how='outer')
features_df = features_df.merge(homework_by_week, left_on=["username", "week"], right_on=["username", "week"], how='outer')
#features_df = features_df.merge(mapleta_by_week, left_on=["username", "week"], right_on=["username", "week"], how='outer')
features_df = features_df.merge(quiz_by_week, left_on=["username", "week"], right_on=["username", "week"], how='outer')
features_df = features_df.query('week >= 1 & week <= 18')



#補值，以0取代-1與nan，並捨棄不需要的欄位
features_df = features_df.replace([-1], [0]) #-1 => 0
features_df = features_df.fillna(0) #NaN => 0
features_df = features_df.drop('email', 1)
features_df = features_df.drop('course_id', 1)
features_df = features_df.drop('label', 1)

features_df.to_csv('result/all_features_by_week.csv', index=False)

