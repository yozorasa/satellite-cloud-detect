// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.05; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, Mat& binYOLO, Mat& binAll, const vector<Mat>& outs);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, Mat& binYOLO, Mat& binAll);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

int main(int argc, char** argv)
{
    /*CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
    parser.printMessage();
    return 0;
    }*/
    // Load names of classes
    string classesFile = "config/cloud.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "config/yolov3-cloud.cfg";
    String modelWeights = "config/yolov3-cloud_20000.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    //net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);
    //net.setPreferableTarget(DNN_TARGET_OPENCL);

    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob, binAll, binYOLO;

    try {

        //outputFile = "yolo_out_cpp.avi";
        //if (parser.has("image"))
        //{
        // Open the image file
        //str = parser.get<String>("image");
        str = "0.jpg";
        ifstream ifile(str);
        if (!ifile) throw("error");
        cap.open(str);
        //str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.jpg");
        outputFile = str;
        /*}
        else if (parser.has("video"))
        {
        // Open the video file
        str = parser.get<String>("video");
        ifstream ifile(str);
        if (!ifile) throw("error");
        cap.open(str);
        //str.replace(str.end() - 4, str.end(), "_yolo_out_cpp.avi");
        str.replace(str.end() - 4, str.end(), "_predict.avi");
        outputFile = str;
        }
        // Open the webcaom
        else cap.open(parser.get<int>("device"));
        */
    }
    catch (...) {
        cout << "Could not open the input image/video stream" << endl;
        system("pause");
        return 0;
    }

    // Get the video writer initialized to save the output video
    /*if (!parser.has("image")) {
    video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    }
    */
    // Create a window
    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            //waitKey(3000);
            break;
        }

        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
        //blob = blobFromImage(frame, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

        //Sets the input to the network
        net.setInput(blob);

        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));

        binAll = frame.clone();
        cvtColor(frame, binAll, COLOR_BGR2GRAY);
        binYOLO = binAll.clone();
        binYOLO.setTo(Scalar::all(0));

        for (int i = 0; i < binAll.rows; i++) {
            for (int j = 0; j < binAll.cols; j++) {
                int idata = binAll.at<uchar>(i, j);
                if (idata < 140)
                    binAll.at<uchar>(i, j) = 0;
            }
        }
        // Remove the bounding boxes with low confidence
        postprocess(frame, binYOLO, binAll, outs);

        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        string label = format("Inference time for a frame : %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

        // Write the frame with the detection boxes
        //Mat detectedFrame;
        //frame.convertTo(detectedFrame, CV_8U);
        //if (parser.has("image")) imwrite(outputFile, detectedFrame);
        //else video.write(detectedFrame);
        //imwrite(outputFile, detectedFrame);
        imwrite("rect.jpg", frame);
        imwrite("binary.jpg", binYOLO);
        //imshow(kWinName, frame);
        /*imshow("rect", frame);
        imshow("binAll", binAll);
        imshow("binYOLO", binYOLO);
        */
    }

    cap.release();
    //if (!parser.has("image")) video.release();
    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, Mat& binYOLO, Mat& binAll, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int right = box.x + box.width;
        int bottom = box.y + box.height;
        if (left < 0)
            left = 0;
        if (top < 0)
            top = 0;
        if (right > frame.cols)
            right = frame.cols;
        if (bottom > frame.rows)
            bottom = frame.rows;
        drawPred(classIds[idx], confidences[idx], left, top,
            right, bottom, frame, binYOLO, binAll);
    }
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, Mat& binYOLO, Mat& binAll)
{
    //cout << endl;
    //cout << left << " " << right << endl;
    //cout << top << " " << bottom << endl;
    for (int i = top; i < bottom; i++)
        for (int j = left; j < right; j++)
            binYOLO.at<uchar>(i, j) = binAll.at<uchar>(i, j);
            
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    //rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    //putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}