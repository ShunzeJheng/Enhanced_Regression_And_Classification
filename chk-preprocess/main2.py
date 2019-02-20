# coding: utf-8
#程式目的:產生各個週次的資料集
#在產生a班或b班的資料前，記得改資料夾或資料集的名稱，如class_a_csv或class_b_csv

import pandas as pd

#讀取各個資料表
all_features_by_week = pd.read_csv('result_a/all_features_by_week.csv', sep=',')
all_features_by_week = all_features_by_week.drop('tmp', axis=1)

#讀取學期成績 for regression
final_score = pd.read_csv('class_a_csv/final_score.csv', sep=',')
#讀取學期成績 for classificaion，分成4類
#final_score = pd.read_csv('class_a_csv/final_score_4_label.csv', sep=',')

#需要使用indicator這個欄位時，可以將註解拿掉，對應到line 56與line 75
#indicator = pd.read_csv('class_a_csv/indicator.csv', sep=',')

#原始的資料集中可能會包含助教、工作人員的資料，所以藉由final_score中的學生名單取出學生資料，對應到line 59、60與line 78、79
username_list = final_score['username'].values
not_in_username_list = []
all_features_by_week_username = all_features_by_week['username'].values
for username in all_features_by_week_username:
    if username not in username_list:
        not_in_username_list.append(username)

#捨棄num_days欄位，因為它與active_num_days基本上相同
all_features_by_week = all_features_by_week.drop('num_days', axis=1)
#捨棄指定的欄位
"""
all_features_by_week = all_features_by_week.drop('forum_avg_count', axis=1)
all_features_by_week = all_features_by_week.drop('forum_num_days', axis=1)
all_features_by_week = all_features_by_week.drop('forum_sum_count', axis=1)
all_features_by_week = all_features_by_week.drop('incomplete_rate', axis=1)
all_features_by_week = all_features_by_week.drop('video_forward_seek_avg', axis=1)
all_features_by_week = all_features_by_week.drop('video_backward_seek_avg', axis=1)
all_features_by_week = all_features_by_week.drop('video_stop_backward_seek_avg', axis=1)
"""

#產生16個不同週次的資料集
week_order = ['1-2', '1-3', '1-4', '1-5', '1-6', '2-3', '2-4', '2-5', '2-6', '3-4', '3-5', '3-6', '4-5', '4-6', '5-6']
week_num = 0
for i in range(1,6):
    for j in range(1,7):
        features_df = all_features_by_week
        if i < j:
            hw_mean = pd.read_csv('hw_mean_a/hw_mean_' + week_order[week_num] + '.csv', sep=',')
            exam_by_week = pd.read_csv('exam_score_a/exam_score_' + week_order[week_num] + '.csv', sep=',')

            features_df = features_df.query('week >= ' + str(i) + ' & week <= ' + str(j))
            features_df = features_df.groupby(['username'], as_index=False).mean()
            features_df = features_df.drop('week', 1)

            features_df = features_df.merge(hw_mean, left_on=["username"], right_on=["username"], how='left')
            features_df = features_df.merge(exam_by_week, left_on=["username"], right_on=["username"], how='left')
            #features_df = features_df.merge(indicator, left_on=["username"], right_on=["username"], how='left')
            features_df = features_df.merge(final_score, left_on=["username"], right_on=["username"], how='left')

            for username in not_in_username_list:
                features_df = features_df[features_df.username != username]

            features_df.rename(columns={'exam_mean': "qz_mean"}, inplace=True)
            features_df.to_csv('result_a/ncu_data_week_' + week_order[week_num] + '.csv', index=False)
            week_num += 1
        if i == 1 and j == 1:
            hw_mean = pd.read_csv('hw_mean_a/hw_mean_1-1.csv', sep=',')
            exam_by_week = pd.read_csv('exam_score_a/exam_score_1-1.csv', sep=',')

            features_df = features_df.query('week >= ' + str(i) + ' & week <= ' + str(j))
            features_df = features_df.groupby(['username'], as_index=False).mean()
            features_df = features_df.drop('week', 1)

            features_df = features_df.merge(hw_mean, left_on=["username"], right_on=["username"], how='left')
            features_df = features_df.merge(exam_by_week, left_on=["username"], right_on=["username"], how='left')
            #features_df = features_df.merge(indicator, left_on=["username"], right_on=["username"], how='left')
            features_df = features_df.merge(final_score, left_on=["username"], right_on=["username"], how='left')

            for username in not_in_username_list:
                features_df = features_df[features_df.username != username]

            features_df.rename(columns={'exam_mean': "qz_mean"}, inplace=True)
            features_df.to_csv('result_a/ncu_data_week_1-1.csv', index=False)