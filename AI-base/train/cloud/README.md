clone https://github.com/yozorasa/satellite-cloud-detect.git

## YOLO Train

1. clone https://github.com/AlexeyAB/darknet.git
2. 以 visual studio 打開 clone 後的 darknet/build/darknet/**darknet.sln**
3. 建置 darknet (需要使用GPU，詳細建置方法請參考https://github.com/AlexeyAB/darknet#how-to-compile-on-windows)
4. 將此專案的 satellite-cloud-detect/train/cloud/ 資料夾 複製到darknet/build/darknet/x64/
5. (更改 training set) 替換 darknet/build/darknet/x64/cloud/images/
6. (更改 training set) 替換 darknet/build/darknet/x64/cloud/labels/
7. (更改 training set) 執行 darknet/build/darknet/x64/cloud/**divideTestTrainSet.py** 自動生成新的 **test.txt** 和 **train.txt**
8. 執行 darknet/build/darknet/x64/cloud/**train.bat** 開始訓練YOLO
9. 訓練結果 每100次迭代後的 weight 將會 save 至 darknet/build/darknet/x64/backup/ 資料夾
10. 若需要使用已經訓練過過的 weight 檔案繼續進行訓練，請將 **train.bat** 中的 **cloud/darknet53.conv.74** 替換成 weight 檔案的路徑

        若不需要更改 training set 請跳過 步驟5.~步驟7.
