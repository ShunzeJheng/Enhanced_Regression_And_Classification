import matplotlib.pyplot as plt

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


datasets = pd.read_csv('result/data_with_predict_blended.csv', sep=',')


features_val = datasets[datasets.evaluation_num == 1].values


rank = pd.read_csv('result/data_with_standard.csv', sep=',')
username = rank['username'].values
first_part = username[0:12]
second_part = username[12:27]
third_part = username[27:42]
fourth_part = username[42:54]

real_score = []
predict_score = []

for row in features_val:
    if row[2] in first_part:
        real_score.append(row[9])
        predict_score.append(row[8])
for row in features_val:
    if row[2] in second_part:
        real_score.append(row[9])
        predict_score.append(row[8])
for row in features_val:
    if row[2] in third_part:
        real_score.append(row[9])
        predict_score.append(row[8])
for row in features_val:
    if row[2] in fourth_part:
        real_score.append(row[9])
        predict_score.append(row[8])

plt.scatter(real_score[0:12], predict_score[0:12], c='b', label = 'top 25%')
plt.scatter(real_score[12:27], predict_score[12:27], c='g', label = '25%-50%')
plt.scatter(real_score[27:42], predict_score[27:42], c='y', label = '50%-75%')
plt.scatter(real_score[42:54], predict_score[42:54], c='r', label = 'last 25%')
plt.title('without_outliers')
plt.xlabel('real_score')
plt.ylabel('predicted_score')
plt.legend()
plt.savefig('result/real_score_to_predicted_score.png')
plt.clf()

real_score = []
residual = []

for row in features_val:
    if row[2] in first_part:
        real_score.append(row[9])
        residual.append(row[8] - row[9])
for row in features_val:
    if row[2] in second_part:
        real_score.append(row[9])
        residual.append(row[8] - row[9])
for row in features_val:
    if row[2] in third_part:
        real_score.append(row[9])
        residual.append(row[8] - row[9])
for row in features_val:
    if row[2] in fourth_part:
        real_score.append(row[9])
        residual.append(row[8] - row[9])


plt.scatter(real_score[0:12], residual[0:12], c='b', label = 'top 25%')
plt.scatter(real_score[12:27], residual[12:27], c='g', label = '25%-50%')
plt.scatter(real_score[27:42], residual[27:42], c='y', label = '50%-75%')
plt.scatter(real_score[42:54], residual[42:54], c='r', label = 'last 25%')
plt.title('without_outliers')
plt.xlabel('real_score')
plt.ylabel('residual')
plt.legend()
plt.savefig('result/real_score_to_residual.png')
plt.clf()
"""
plt.scatter(real_score, predicted_score, c='b')
plt.scatter(real_score, residual, c='r')
plt.xlabel('real_score')
plt.ylabel('predicted_score_and_residual')
plt.savefig('result/real_score_to_predicted_score_and_residual.png')
plt.clf()
"""