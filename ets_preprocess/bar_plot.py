# coding: utf-8
#程式目的:產生學期成績的bar chart

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('result/ncu_data_week_1-6(1a).csv', sep=',')

score_str = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '101-110']
score_num = []
for i in range(1,12):
    if i == 1:
        final_score = dataset[dataset.final_score <= i*10]
        final_score = final_score[final_score.final_score >= (i-1)*10]
        score_num.append(len(final_score))
    else:
        final_score = dataset[dataset.final_score <= i * 10]
        final_score = final_score[final_score.final_score > (i - 1) * 10]
        score_num.append(len(final_score))
print score_num
print len(score_num)
xs = [i + 0.1 for i, _ in enumerate(score_str)]
plt.bar(xs, score_num)
plt.title('ets bar plot')
plt.xlabel('score')
plt.ylabel('count')
plt.bar(xs, score_num, color='blue')
plt.xticks([i + 0.1 for i, _ in enumerate(score_str)], score_str)
plt.savefig('result/bar_plot.png')