clone https://github.com/yozorasa/satellite-cloud-detect.git

## YOLO Test

1. 以 opencv c++ 編譯 satellite-cloud-detect/test/**yolov3.cpp**
2. 將 編譯完成的 yolov3.exe 放置至 satellite-cloud-detect/test/ 資料夾 (替代原本的yolov3.exe)
3. 命名 需預測的衛星影像圖片 為 **0.jpg** (替代原本的0.jpg)
4. 執行 **yolov3.exe**
5. 執行完畢後 得到預測結果 **rect.jpg** 和 **binary.jpg**

        編譯 yolov3.cpp 時 請務必使用 opencv3.4.3
