chk_generate_indicator
=====================
賦予高三增能微積分A、B班資料點指標函數。


## Dependency
- Python 2.7.13
- Anaconda2


## 資料夾說明

- all_data_a and all_data_b資料夾中含有原始資料。
- all_result_a and all_result_b資料夾中含有輸出的資料集。
- classifier_result_a and classifier_result_b 資料夾中含有分類的準確度。


## 執行順序(以Decision Tree為例，其中演算法依此類推)

1. 執行classifier_each_feature_dt.py 取得各種參數與特徵組合的Decision Tree分類準確度。
2. 觀察分類結果得到想要的參數後，執行 apply_classifier_to_append_indicator.py並修改參數，取得具有指標函數的資料集。

## 補充
- 參數的選擇通常是選擇含有全部的特徵組合的資料集加上最準確的參數組合。

## Authors
- 鄭舜澤