# satellite-cloud-detect
        此專案以 Windows 做為環境建置
Satellite Cloud Image Source From https://discover.digitalglobe.com/
clone https://github.com/yozorasa/satellite-cloud-detect.git

## YOLO Test

1. 以 opencv c++ 編譯 satellite-cloud-detect/AI-base/test/**yolov3.cpp**
2. 將 編譯完成的 yolov3.exe 放置至 satellite-cloud-detect/AI-base/test/ 資料夾 (替代原本的yolov3.exe)
3. 命名 需預測的衛星影像圖片 為 **0.jpg** (替代原本的0.jpg)
4. 執行 **yolov3.exe**
5. 執行完畢後 得到預測結果 **rect.jpg** 和 **binary.jpg**

        編譯 yolov3.cpp 時 請務必使用 opencv3.4.3


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



## Tradition
皆以 OpenCV C++ 環境編譯

### grabCut_only.cpp

- 簡介: 只有GrabCut 的雲層偵測方法
- 參數設置: 
  - whiteRGB: 雲層二值化的灰度門檻值(範圍: 0 - 255)
  - filePlace: 輸出檔案儲存位置，路徑(ex: satellite-cloud-detect/Tradition/grabCut_only/example_result)
  - srcfilePlace: 衛星影像來原檔案位置，路徑(ex: satellite-cloud-detect/Tradition/grabCut_only/example_inputImage)
  - fIndex: 要預測的衛星影像編號
- 使用步驟:
  1. 待預測的衛星影像重新命名為 **0.jpg**、**1.jpg**、**2.jpg**、、、
  2. 建立目標資料夾儲存輸出影像，資料夾內新增 **img**、**lbp**、**hull** 三個空資料夾
  3. 檔案路徑設置(filePlace、srcfilePlace)，其餘參數視需要修改
  4. 以 openCV C++ 環境編譯並執行
- 輸出結果: 
  - lbp 資料夾: 各別雲層ROI 與 LBP轉換結果，做為SVM Training Set
  - hull 資料夾: 衛星影像偵測出雲層位置後，經hull簡化輪廓的二值化影像
  - img 資料夾:
    1. 包含圈選輪廓之灰階影像
    2. 未經雲層大小篩選之二值化雲層偵測影像
    3. 經雲層大小篩選，且標示出各雲層區域ROI
    4. 洋紅為底之前景影像


### grabCutAndSVM.cpp

- 簡介: 包含SVM Training，與GrabCut雲層偵測後以SVM Test 分類前景是否為雲層
- 參數設置: 
  - record: SVM 訓練模型儲存路徑(ex: satellite-cloud-detect/Tradition/grabCutAndSVM/example_result/)
  - loadLocationC: 雲層LBP 影像訓練資料集影像路徑(ex: satellite-cloud-detect/Tradition/grabCutAndSVM/example_SVMTrainingSet/cloud/)
  - loadLocationO: 非雲層LBP 影像訓練資料集影像路徑(ex: satellite-cloud-detect/Tradition/grabCutAndSVM/example_SVMTrainingSet/notCloud/)
  - svmFileName: SVM Training Model 儲存檔案名稱
  - cloudAmount: 雲層Training Data 數量
  - otherAmount: 非雲層Training Data 數量
  - whiteRGB: 雲層二值化的灰度門檻值(範圍: 0 - 255)
  - filePlace: 輸出檔案儲存位置，路徑(ex: satellite-cloud-detect/Tradition/grabCutAndSVM/example_result)
  - srcfilePlace: 衛星影像來原檔案位置，路徑(ex: satellite-cloud-detect/Tradition/grabCutAndSVM/example_inputImage/)
  - fIndex: 要預測的衛星影像編號
- 使用步驟:
  1. 人工分類LBP影像，分成雲與非雲兩個資料夾，影像各自重新命名為 **0_lbp.jpg**、**1_lbp.jpg**、**2_lbp.jpg**、、、
  2. 建立目標資料夾儲存輸出影像，資料夾內新增 **img**、**lbp** 兩個空資料夾
  3. 檔案路徑設置 (record、loadLocationC、loadLocationO、filePlace、srcfilePlace)
  4. SVM Training 數量(cloudAmount、otherAmount)，**histogramCal與tag參數的陣列維度也要同步修改**
  5. 以 openCV C++ 環境編譯並執行
- 輸出結果: 
  - lbp 資料夾: 各別雲層ROI 與 LBP轉換結果，做為SVM Training Set
  - img 資料夾:
    1. origin: 包含圈選輪廓之原圖影像
    2. beforeFilter: 未經雲層大小篩選之二值化雲層偵測影像
    3. onlySizeFilter: 經雲層大小篩選之二值化雲層偵測影像
    4. afterSVMFilter: 經SVM Test篩選之二值化雲層偵測影像


