#include "opencv2/opencv.hpp"
#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace ml;

// ============================================================ //
// ================== SVM Training Setting ==================== //
// ============================================================ //
String record = "<LBP training Set file path>";
String loadLocationC = "<LBP training Set file path (Cloud)>";
String loadLocationO = "<LBP training Set file path (Not Cloud)>";
String fileType = ".jpg";
String svmFileName = "<svm model name>.xml";
int cloudAmount = 504;
int otherAmount = 252;
float histTemp[256] = { 0 };
float histogramCal[504 + 252][256] = { 0 };
int tag[504 + 252] = { 0 };

HOGDescriptor *hog = new HOGDescriptor(Size(64, 64), Size(8, 8), Size(4, 4), Size(4, 4), 9, 1);

vector<Mat> hogDatas;

void convert_to_ml(Mat &trainData)
{
	//--Convert data
	const int rows = (int)hogDatas.size();
	const int cols = (int)std::max(hogDatas[0].cols, hogDatas[0].rows);
	Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = Mat(rows, cols, CV_32FC1);

	for (size_t i = 0; i < hogDatas.size(); ++i)
	{
		CV_Assert(hogDatas[i].cols == 1 || hogDatas[i].rows == 1);

		if (hogDatas[i].cols == 1)
		{
			transpose(hogDatas[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (hogDatas[i].rows == 1)
		{
			hogDatas[i].copyTo(trainData.row((int)i));
		}
	}
}

void convert_to_ml2(Mat &trainData, Mat hogDatas)
{
	//--Convert data
	// const int rows = (int)hogDatas.size();
	const int cols = (int)std::max(hogDatas.cols, hogDatas.rows);
	Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = Mat(1, cols, CV_32FC1);
	CV_Assert(hogDatas.cols == 1 || hogDatas.rows == 1);
	if (hogDatas.cols == 1)
	{
		transpose(hogDatas, tmp);
		tmp.copyTo(trainData.row(0));
	}
	else if (hogDatas.rows == 1)
	{
		hogDatas.copyTo(trainData.row(0));
	}
}

vector<float> hogCompute(Mat lbp)
{
	vector<float> hogDescriptors;
	resize(lbp, lbp, Size(64, 64), 0, 0, CV_INTER_AREA);
	hog->compute(lbp, hogDescriptors, Size(1, 1), Size(0, 0));
	hogDatas.push_back(Mat(hogDescriptors).clone());
	return hogDescriptors;
}

void getSVMParams(SVM *svm)
{
	cout << "Kernel type     : " << svm->getKernelType() << endl;
	cout << "Type            : " << svm->getType() << endl;
	cout << "C               : " << svm->getC() << endl;
	cout << "Degree          : " << svm->getDegree() << endl;
	cout << "Nu              : " << svm->getNu() << endl;
	cout << "Gamma           : " << svm->getGamma() << endl;
}

Ptr<SVM> SVMtrain(Mat &trainMat, Mat trainLabels)
{
	Ptr<SVM> svm = SVM::create();
	// svm->setGamma(0.50625);
	// svm->setC(100);
	svm->setKernel(SVM::LINEAR);
	svm->setType(SVM::C_SVC);
	Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
	svm->train(td);
	//svm->trainAuto(td);
	svm->save(record + svmFileName);
	getSVMParams(svm);
	return svm;
}

Mat LBP(Mat src_image)
{
	Mat temp_image = src_image;
	Mat Image(temp_image.rows, temp_image.cols, CV_8UC1);
	Mat lbp(temp_image.rows, temp_image.cols, CV_8UC1);

	if (temp_image.channels() == 3)
		cvtColor(temp_image, Image, CV_BGR2GRAY);
	int center = 0;
	int center_lbp = 0;

	for (int row = 1; row < Image.rows - 1; row++)
	{
		for (int col = 1; col < Image.cols - 1; col++)
		{

			center = Image.at<uchar>(row, col);
			center_lbp = 0;

			if (center <= Image.at<uchar>(row - 1, col - 1))
				center_lbp += 1;

			if (center <= Image.at<uchar>(row - 1, col))
				center_lbp += 2;

			if (center <= Image.at<uchar>(row - 1, col + 1))
				center_lbp += 4;

			if (center <= Image.at<uchar>(row, col - 1))
				center_lbp += 8;

			if (center <= Image.at<uchar>(row, col + 1))
				center_lbp += 16;

			if (center <= Image.at<uchar>(row + 1, col - 1))
				center_lbp += 32;

			if (center <= Image.at<uchar>(row + 1, col))
				center_lbp += 64;

			if (center <= Image.at<uchar>(row + 1, col + 1))
				center_lbp += 128;
			lbp.at<uchar>(row, col) = center_lbp;
		}
	}
	return lbp;
}

int main()
{
	//============================================================== //
	//========================= SVM training ======================= //
	//========================= SVM training ======================= //
	//============================================================== //
	Mat lbp, roi;
	for (int i = 1; i <= cloudAmount; i++)
	{
		//file << loadLocationC+to_string(i)+"_lbp"+fileType << "\n";
		cout << loadLocationC + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationC + to_string(i) + "_lbp" + fileType, 0);
		hogCompute(lbp);
	}
	for (int i = 1; i <= otherAmount; i++)
	{
		//file << loadLocationO+to_string(i)+"_lbp"+fileType << "\n";
		cout << loadLocationO + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationO + to_string(i) + "_lbp" + fileType, 0);
		hogCompute(lbp);
	}
	for (int i = 0; i < cloudAmount + otherAmount; i++)
	{
		if (i < cloudAmount)
			tag[i] = 1;
		else
			tag[i] = -1;
	}
	Mat hogTrain_data;
	convert_to_ml(hogTrain_data);

	const int num_data = cloudAmount + otherAmount; //資料數

	Mat labelsMat(num_data, 1, CV_32SC1, tag);

	Ptr<SVM> svm = SVMtrain(hogTrain_data, labelsMat);

	//==================================================================== //
	//========================= grabCut & SVM test ======================= //
	//========================= grabCut & SVM test ======================= //
	//==================================================================== //

	Mat image;
	Mat rgbImage1;
	int whiteRGB = 130;
	Scalar bgColor = Scalar(255, 0, 255);
	String filePlace = "<result file path>";
	String srcfileType = ".jpg";
	String srcfilePlace = "<source file path>";
	int fIndex = 1;
	const time_t ctt = time(0);
	cout << asctime(localtime(&ctt)) << std::endl;
	while (true)
	{
		String imgName = to_string(fIndex);
		rgbImage1 = cv::imread(srcfilePlace + imgName + srcfileType);
		image = cv::imread(srcfilePlace + imgName + srcfileType, CV_LOAD_IMAGE_GRAYSCALE);

		if (!image.data) // Check for invalid input
		{
			const time_t ctt = time(0);
			cout << asctime(localtime(&ctt)) << std::endl;
			cout << "Could not open or find the image" << std::endl;
			system("pause");
			return -1;
		}
		// namedWindow(imgName, WINDOW_AUTOSIZE);
		// imshow(imgName, image);

		int allArea = image.cols * image.rows;

		// ====================================================================================================//
		// ============================================ GRABCUT ===============================================//
		// ====================================================================================================//

		Mat result(image.size(), CV_8UC1, Scalar(GC_BGD)); // segmentation result (4 possible values) (second)
		Mat mask;										   // segmentation result (4 possible values) (fist)
		Mat bgModel, fgModel;							   // the models (internally used)
		Mat foregroundTemp(image.size(), CV_8UC3, bgColor);

		// ---------------------------------------- 前後景MASK ---------------------------------//
		for (int y = 0; y < image.rows; y++)
		{
			uchar *ptr2 = result.ptr<uchar>(y);
			uchar *ptr1 = image.ptr<uchar>(y);
			for (int x = 0; x < image.cols; x++)
			{
				if (!((int)ptr1[x] > whiteRGB))
				{
					ptr2[x] = GC_PR_BGD;
				}
				else if ((int)ptr1[x] > whiteRGB)
				{
					ptr2[x] = GC_PR_FGD;
				}
			}
		}
		compare(result, GC_PR_FGD, mask, CMP_EQ);
		image.copyTo(foregroundTemp, mask); // foregroundTemp = cloud with bgColor before grabcut
		// ----------------------------------- grabcut ---------------------------------//
		Mat rgbImage;
		cvtColor(image, rgbImage, CV_GRAY2BGR);
		grabCut(rgbImage,				// input image
			result,					// segmentation result
			cv::Rect(),				// rectangle containing foreground
			bgModel, fgModel,		// models
			2,						// number of iterations
			cv::GC_INIT_WITH_MASK); // use rectangle
									// Get the pixels marked as likely foreground

		// ====================================================================================================//
		// ==================================== FIND CLOUD ROI ================================================//
		// ====================================================================================================//

		Mat foreground(image.size(), CV_8UC3, bgColor);
		Mat foregroundBinary(image.size(), CV_8UC1, Scalar(0));
		Mat whiteImg(image.size(), CV_8UC1, Scalar(255));
		Mat getContoursSVM(image.size(), CV_8UC3, Scalar(0,0,0));
		Mat getContours(image.size(), CV_8UC1, Scalar(0));
		// Mat getHulls_b(image.size(), CV_8UC1, Scalar(0));
		Mat afterFilterArea(image.size(), CV_8UC1, Scalar(0));
		Mat imgLBP;

		compare(result, GC_PR_FGD, result, cv::CMP_EQ); // result = mask after grabcut
		rgbImage.copyTo(foreground, result);			// foreground = foregroundTemp after grabcut
		whiteImg.copyTo(foregroundBinary, result);		// foregroundBinary = one channel foreground

														// ----------------------------------- findContours ---------------------------------//
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(foregroundBinary, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		// vector<vector<Point>> hull(contours.size());

		// ----------------------------------- cloud rois ---------------------------------//
		for (int i = 0; i < contours.size(); i++)
		{
			double subArea = contourArea(contours[i], false);
			if (subArea / allArea > 0.0005)
			{
				int contoursIndex = 0;
				Rect bounding_rect = boundingRect(contours[i]);
				Mat foregroundROI = foreground(bounding_rect);
				imgLBP = LBP(foregroundROI);

				imwrite(filePlace + "/lbp/" + imgName + "_" + to_string(i) + "_lbp.jpg", imgLBP);
				imwrite(filePlace + "/lbp/" + imgName + "_" + to_string(i) + "_roi.jpg", foregroundROI);
				drawContours(getContours, contours, i, bgColor, CV_FILLED, 8, hierarchy);

				vector<float> src = hogCompute(imgLBP);
				Mat hogTrain_data;
				convert_to_ml2(hogTrain_data, Mat(src));
				int response = svm->predict(hogTrain_data);
				if (response == -1) continue;

				drawContours(getContoursSVM, contours, i, bgColor, CV_FILLED, 8, hierarchy);
				cv::rectangle(getContoursSVM, bounding_rect, bgColor, 2);
				drawContours(rgbImage1, contours, i, Scalar(0,0,255), 2, 8, hierarchy);
			}
		}
		imwrite(filePlace + "/img/" + imgName + "_1-origin.jpg", rgbImage1);
		imwrite(filePlace + "/img/" + imgName + "_2-beforeFilter.jpg", foregroundBinary);
		imwrite(filePlace + "/img/" + imgName + "_4-afterSVMFilter.jpg", getContoursSVM);
		imwrite(filePlace + "/img/" + imgName + "_3-onlySizeFilter.jpg", getContours);

		fIndex++;
	}

	waitKey();
	return 0;
}