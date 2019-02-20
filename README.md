Enhanced Regression and Classification
=====================
迴歸演算法與分類演算法的改進。
改進後的迴歸演算法為：Classification + Indicator variables + PCR。
改進後的分類演算法為：Resampling + Voting classifier。


## Dependency
- Python 2.7.13
- Anaconda2

## Installation

### 使用PyCharm IDE

1. 到PyCharm官網下載安裝程式，並安裝。
2. 到Anaconda2官網下載安裝程式，並安裝。
3. 打開PyCharm，File -> Open -> 找到任一專案(如chk-preprocess)的資料夾 -> 點選ok即可打開此專案。
4. 打開PyCharm，File -> Setttings -> Project Interpreter -> 齒輪內選Add Local-> 找到安裝好的Anaconda2資料夾 -> 點選右下角apply。
5. 完成以上步驟即可執行專案。

## 專案名稱說明

- ets開頭的專案代表所使用的資料集為「聯合微積分教學-地科班」資料集。
- chk開頭的專案代表所使用的資料集為「高三增能微積分A、B班」資料集。


## Enhanced Regression

### 相關專案

- ets_preprocess
- ets_generate_indicator
- ets_with_Indicator
- chk_preprocess
- chk_generate_indicator
- chk_with_Indicator
- feature_selection

### 流程

以地科班資料集為例

1. 透過ets_preprocess專案產生55個feature的原始資料集。
2. 透過feature_selection專案產生經過person相關係數篩選後的資料集。
3. 透過ets_generate_indicator專案產生附有Indicator variables的資料集。
4. 透過ets_with_Indicator專案產生最終結果。

## Enhanced Classificaion

### 相關專案

- ets_preprocess
- chk_preprocess
- feature_selection
- classificaion_voting

### 流程

以地科班資料集為例

1. 透過ets_preprocess專案產生55個feature的原始資料集。
2. 透過feature_selection專案產生經過Single Variable Classifier篩選後的資料集。
3. 透過classificaion_voting專案產生最終結果(Resampling 與 Voting classifier皆在classificaion_voting專案中)。


## 其他專案補充

### 相關專案

- ets_PCR
- chk_PCR
- ets_regression_algorithm
- chk_regression_algorithm
- ets_classifier
- chk_classifier

### 用途

- ets_PCR與chk_PCR專案目的為執行原先的PCR方法。
- ets_regression_algorithm與chk_regression_algorithm專案目的為執行多種迴歸演算法，包括以下演算法：

```
Multiple Linear Regression(MLR)
Classification And Regression Tree(CART)
Quantile Regression
Robust Regression
Support Vector Regression(SVR)
Principle Component Regression(PCR)
```

- ets_classifier與chk_classifier專案目的為執行多種分類演算法，包括以下演算法：

```
Gaussian Naive Bayes(GaNB)
Support Vector Machine
Logistic Regression
Decision Tree
Random Forest
Neural Network
```

## Authors
- 鄭舜澤