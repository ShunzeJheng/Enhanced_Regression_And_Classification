# coding: utf-8
#程式目的:產生各個週次的資料集

import pandas as pd

#讀取各個資料表
all_features_by_week = pd.read_csv('result/all_features_by_week.csv', sep=',')

#after_school_counseling.csv是紀錄after_school_counseling這個欄位的資料表，但後來因為高三增能沒有這個欄位，所以不使用，對應到line 48
#after_school_counseling = pd.read_csv('data/after_school_counseling.csv', sep=',')

#需要使用indicator這個欄位時，可以將註解拿掉，對應到line 49
#indicator = pd.read_csv('data/indicator.csv', sep=',')

#讀取學期成績 for regression
final_score = pd.read_csv('data/final_score.csv', sep=',')
#讀取學期成績 for classificaion，分成4類
#final_score = pd.read_csv('data/final_score_4_label.csv', sep=',')


#原始的資料集中可能會包含助教、工作人員的資料，所以藉由final_score中的學生名單取出學生資料，對應到line 52、53
username_list = final_score['username'].values
not_in_username_list = []
all_features_by_week_username = all_features_by_week['username'].values
all_features_by_week_username = list(set(all_features_by_week_username))
for username in all_features_by_week_username:
    if username not in username_list:
        not_in_username_list.append(username)

#若只想要取特定幾個特徵，可以在features_list中輸入指定要取的特徵，對應到line 42
#features_list = ['username', 'week', 'active_num_days', 'active_sum_count', 'mt_practice_sum', 'mt_unit_sum'
#    , 'mt_online_num_day', 'mt_online_practice_num_day', 'hw_mean', 'qz_mean']

#捨棄num_days欄位，因為它與active_num_days基本上相同
all_features_by_week = all_features_by_week.drop('num_days', axis=1)

#產生5個不同週次的資料集
filename_list = ['1-6(1a)', '1-12(2a)', '1-18(3a)', '7-12(2d)', '13-18(3d)']
query_list = ['week >= 1 & week <= 6', 'week >= 1 & week <= 12', 'week >= 1 & week <= 18', 'week >= 7 & week <= 12', 'week >= 13 & week <= 18']
for i in range(0,5):
    features_df = all_features_by_week
    #features_df = all_features_by_week[features_list]
    features_df = features_df.query(query_list[i])

    features_df = features_df.groupby(['username'], as_index=False).mean()
    features_df = features_df.drop('week', 1)

    #features_df = features_df.merge(after_school_counseling, left_on=["username"], right_on=["username"], how='left')
    #features_df = features_df.merge(indicator, left_on=["username"], right_on=["username"], how='left')
    features_df = features_df.merge(final_score, left_on=["username"], right_on=["username"], how='left')

    for username in not_in_username_list:
        features_df = features_df[features_df.username != username]

    #將104602004、104602015這兩位學生的資料去除，這兩位學生休學，所以不計入資料集
    features_df = features_df[features_df.username != 104602004]
    features_df = features_df[features_df.username != 104602015]

    features_df.to_csv('result/ncu_data_week_' + filename_list[i] + '.csv', index=False)