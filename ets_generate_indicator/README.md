chk_generate_indicator
=====================
賦予微積分地科班資料點指標函數。


## Dependency
- Python 2.7.13
- Anaconda2


## 資料夾說明

- data資料夾中含有原始資料。
- result資料夾中含有輸出的資料集。
- classifier_result資料夾中含有分類的準確度。


## 執行順序(以Decision Tree為例，其中演算法依此類推)

1. 執行 classifier_each_feature_dt.py取得各種參數與特徵組合的Decision Tree分類準確度。
2. 觀察分類結果得到想要的參數後，執行 apply_classifier_to_append_indicator.py並修改參數，取得具有指標函數的資料集。

## 補充
- 參數的選擇通常是選擇含有全部的特徵組合的資料集加上最準確的參數組合。

## Authors
- 鄭舜澤