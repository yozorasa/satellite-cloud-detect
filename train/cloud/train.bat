@ECHO ON
cd ..
darknet.exe detector train cloud/cloud.data cloud/yolov3-cloud.cfg cloud/darknet53.conv.74 -dont_show
pause