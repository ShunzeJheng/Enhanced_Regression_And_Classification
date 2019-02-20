feature_selection
=====================
進行特徵選取並輸出資料集。


## Dependency
- Python 2.7.13
- Anaconda2

## 資料夾名稱說明

- ets開頭的專案代表所使用的資料集為「聯合微積分教學-地科班」資料集。
- chk開頭的專案代表所使用的資料集為「高三增能微積分A、B班」資料集。

## 資料夾說明

- data 結尾的資料夾中含有原始資料。
- result 結尾的資料夾中含有輸出的結果。


## 執行順序(以地科班資料集為例)

### Pearson correlation coefficient for regression 

1. 執行 ets_person.py取得皮爾森相關係數。
2. 執行 ets_pearson_get_dataset.py取得特徵選取後的資料集。

### Single Variable Classifer for classification

1. 執行 ets_single_variable_classifier.py取得各個特徵的AUC。
2. 執行 ets_classifier_get_dataset.py取得特徵選取後的資料集。

## Authors
- 鄭舜澤