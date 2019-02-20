import numpy as np
import statsmodels.api as sm
import pylab

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import preprocessing


datasets = pd.read_csv('cross_validation_Residual_table_blended.csv', sep=',')

features_header = list(datasets)[0:4]
features_val = datasets[datasets.evaluation_num == 1]
residual = features_val[features_header[2]].values - features_val[features_header[3]].values
print residual

residual_str = ""
for item in residual:
    residual_str = residual_str + "," + str(item)
print residual_str

sm.qqplot(residual, line='45' , fit=True)
pylab.show()