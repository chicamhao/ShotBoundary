#define _CRT_SECURE_NO_DEPRECATE
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "header.h"
#include <algorithm>

using namespace std;
using namespace cv;
using namespace rapidxml;

//// hyperparameters
int threhold_detected_frame = 15;
int threhold_false_frame = 30;
int threhold_missing_frame = 60;

int threshold_type = 1;
int numberOfVideo = 10;
int threshold_value_msd = 26;
int const max_value_msd = 60;
int const numberOfDetectedGroundTruth = 10;
int const max_value = 100;
int const max_type = 4;
int threshold_value = 16;
int const max_BINARY_value = 255;
bool DebugMode = false;

//record frames, key: frame's number, value: weight
std::map<int, int> color_recordFrames;
std::map<int, int> histogram_recordFrames;
std::map<int, int> edge_recordFrames;
std::map<int, int> block_recordFrames;

// weight vectors
vector<double> msdVector1, msdVector2, msdVector3, msdVector4;

// tempory for right gradient
vector<double> rightGradientVector;
// Methods
void DrawHist11(int, int, Mat, int);
void DrawChart(vector<double>,String);
int SearchMax(vector<double>);
void detectBlob(Mat& srcImage, vector<Point>& featurePts, double threhod, uchar nKernel);
Mat getLoG(const Mat& srcImage, uchar ksize, float sigma);
float calculatePrecision(std::map<int, int> detected, vector<int> ground_truth);
float calculateRecall(std::map<int, int> detected, vector<int> ground_truth);

//histogram parameter
double tmpMsd = 0;
int h_bins = 50, s_bins = 60;
int histSize[] = { h_bins, s_bins };
// hue varies from 0 to 179, saturation from 0 to 255
float h_ranges[] = { 0, 180 };
float s_ranges[] = { 0, 256 };
const float* ranges[] = { h_ranges, s_ranges };
// Use the 0-th and 1-st channels
int channels[] = { 0, 1 };

//edge parameter
int dilation_size = 0;
Mat c_frameEdge, p_frameEdge;
Mat c_delate, p_delate;
Mat c_invert, p_invert;
Mat element = getStructuringElement(MORPH_RECT,
	Size(2 * dilation_size + 1, 2 * dilation_size + 1),
	Point(dilation_size, dilation_size));

//block-based parameter
int block_threshold = 5;
int upper_threhold = 100;
vector<Point> currentFeaturePts;
vector<Point> afterFeaturePts;
const float k2 = 1.414f;

void main() 
{
	for (int i = 0; i < 10; ++i) 
	{
		string iString = to_string(i + 1);
		auto pathName = "datasets/" + iString + ".mp4";
		runProgramX(pathName, i + 1);
	}
}

int runProgramX(String path, int index) 
{
	VideoCapture cap(path);

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "ERROR: Cannot open the video file" << endl;
	}

	namedWindow("MainVids", WINDOW_NORMAL); //create a window called "MyVideo"

	Mat frameColor;
	Mat frameGray1, frameGray2;
	Mat frameHSV1, frameHSV2;
	Mat diffFrame;
	Mat hist_1, hist_2;

	bool bSuccess1 = cap.read(frameColor); // read a new frame from video
	if (!bSuccess1) //if not success, break loop
	{
		cout << "ERROR: Cannot read a frame from video file" << endl;
	}
	int height = frameColor.rows;
	int width = frameColor.cols;

	//imageDiffArray
	imd = new double*[height ];
	for (int i = 0; i < height ; ++i)
		imd[i] = new double[width];

	double maxValueAtLogoTransition=0;
	bool startTracking = false;
	int lastLogoFrameId = 0;

	while (1)
	{
		double frameNo = cap.get(CAP_PROP_POS_FRAMES);

		if (!cap.read(frameColor))
		{
			cout << "ERROR: Cannot read a frame from video file" << endl;
			break;
		}
		cvtColor(frameColor, frameGray1, COLOR_BGR2GRAY);
		cvtColor(frameColor, frameHSV1, COLOR_BGR2HSV);
		
		if (!cap.read(frameColor))
		{
			cout << "ERROR: Cannot read a frame from video file" << endl;
			break;
		}

		cvtColor(frameColor, frameGray2, COLOR_BGR2GRAY);
		cvtColor(frameColor, frameHSV2, COLOR_BGR2HSV);

		//----------------------PIXEL COMPARISION---------------------------------------------------------
		double total = 0;
		for (size_t i = 0; i < height; i++)
		{
			for (size_t j = 0; j < width; j++)
			{
				auto diffVal = (int)frameGray1.at<uchar>(i , j ) - (int)frameGray2.at<uchar>(i, j);
				diffVal = diffVal*diffVal;
				imd[i][j] = diffVal;
				total += diffVal;
			}
		}

		double MSD = sqrt(  total / frameColor.total()) ;
		tmpMsd += MSD;
		color_recordFrames.insert(std::pair<int, int>(frameNo, MSD));
		//cout << "PIXEL CAMPARISION: Weight: " << MSD << ", frame: " << frameNo << endl;
		msdVector1.push_back(MSD);

		
		//----------------------HISTOGRAM---------------------------------------------------------
		calcHist(&frameHSV1, 1, channels, Mat(), hist_1, 2, histSize, ranges, true, false);
		normalize(hist_1, hist_1, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&frameHSV2, 1, channels, Mat(), hist_2, 2, histSize, ranges, true, false);
		normalize(hist_2, hist_2, 0, 1, NORM_MINMAX, -1, Mat());

		//using chi-square method
		double weight = compareHist(hist_1, hist_2, 2);
		//cout << "HISTOGRAM: Weight: " << weight << ", frame: " << frameNo << endl;
		histogram_recordFrames.insert(std::pair<int, int>(frameNo, weight));
		msdVector2.push_back(weight);

		//-----------------------EDGE CHANGE RATIO-------------------------------------------------
		vector<Point> myedges1;
		vector<Point> myedges2;

		Canny(c_frameGray, c_frameEdge, 50, 150, 3, false);
		dilate(c_frameEdge, c_delate, element);
		auto invert = (255 - c_delate);
		auto c_and = (c_frameEdge & invert);
		auto c_pixel_sum = sum(c_frameEdge)[0];
		auto out_pixels = sum(c_and)[0];

		Canny(frameGray2, p_frameEdge, 50, 150, 3, false);
		dilate(p_frameEdge, p_delate, element);
		auto p_invert = (255 - p_delate);
		auto p_and = (p_frameEdge & p_invert);
		auto p_pixel_sum = sum(p_frameEdge)[0];
		auto in_pixels = sum(p_and)[0];

		auto edgeWeight = max(in_pixels / p_pixel_sum, out_pixels / c_pixel_sum);
		cout << "EDGE CHANGE RATIO: Weight: " << edgeWeight << ", frame: " << frameNo << endl;
		edge_recordFrames.insert(std::pair<int, int>(frameNo, edgeWeight));
		msdVector3.push_back(edgeWeight);

		//-----------------------BLOCK-BASED-------------------------------------------------
		currentFeaturePts.clear();
		afterFeaturePts.clear();

		detectBlob(Mat& c_frameGray, vector<Point>& currentFeaturePts, block_threshold, 3);
		detectBlob(Mat& c_frameGray2, vector<Point>& afterFeaturePts, block_threshold, 3);

		double totalDifferencePointWeight = 0;
		for (int i = 0; i < currentFeaturePts.size() - 1; ++i)
		{
			for (int j = 0; j < afterFeaturePts.size() - 1; ++j)
			{
				var differencePointWeight =  abs(currentFeaturePts[i].x - afterFeaturePts[i].x) + abs(currentFeaturePts[i].y - afterFeaturePts[i].y);
				totalDifferencePointWeight += differencePointWeight;
			}
		}
		//the lower the difference point weight is, the closer block point detection, the higher weight
		weight = upper_threhold - totalDifferencePointWeight;

		//cout << "HISTOGRAM: Weight: " << weight << ", frame: " << frameNo << endl;
		block_recordFrames.insert(std::pair<int, int>(frameNo, weight));
		msdVector4.push_back(weight);

		//-----------------------DRAW-CHART----------------------------------------------------
		DrawChart(msdVector1,"1");
		DrawChart(msdVector2, "2");
		DrawChart(msdVector3, "3");
		DrawChart(msdVector4, "4");

		imshow("18120353", frameColor); //show the frame in "MyVideo" window

		if (waitKey(50) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	// When everything done, release the video capture object
	cap.release();

	// Declare vector of pairs, stored frame's number and weight of each method.
	vector<pair<int, int> > A;
	vector<pair<int, int> > B;
	vector<pair<int, int> > C;
	vector<pair<int, int> > D;
	vector<int> ground_truth;

	// Copy key-value pair from Map
	// to vector of pairs
	for (auto& it : color_recordFrames) {
		A.push_back(it);
	}
	for (auto& it : histogram_recordFrames) {
		B.push_back(it);
	}
	for (auto& it : edge_recordFrames) {
		C.push_back(it);
	}
	for (auto& it : edge_recordFrames) {
		D.push_back(it);
	}

	// Sort using comparator function
	sort(A.begin(), A.end(), [](const pair<int, int>& left, const pair<int, int>& right) {return left.second > right.second; });
	sort(B.begin(), B.end(), [](const pair<int, int>& left, const pair<int, int>& right) {return left.second > right.second; });
	sort(C.begin(), C.end(), [](const pair<int, int>& left, const pair<int, int>& right) {return left.second > right.second; });
	sort(D.begin(), D.end(), [](const pair<int, int>& left, const pair<int, int>& right) {return left.second > right.second; });

	// Print the sorted value
	int count = 0;
	ofstream color, hist, edge, block;
	color.open("output/color_output.txt");
	hist.open("output/histogram_output.txt");
	edge.open("output/edge_output.txt");
	block.open("output/block_output.txt");

	for (auto& it : A) {

		color << "Color: frame: " << it.first << ", weight :"
			<< it.second << endl;

		if (count == 30) break;
	}
	for (auto& it : B) {

		hist << "Histogram: frame: " << it.first << ", weight :"
			<< it.second << endl;

		if (count == 30) break;
	}
	for (auto& it : C) {

		edge << "Edge: frame: " << it.first << ", weight :"
			<< it.second << endl;

		if (count == 30) break;
	}

	for (auto& it : D) {

		block << "Edge: frame: " << it.first << ", weight :"
			<< it.second << endl;

		if (count == 30) break;
	}

	//read ground truth file
	ofstream ground_truth;
	string iString = to_string(i + 1);
	auto pathName = "groundtruth/" + iString + ".txt";
	ground_truth.open(pathName);

	//get all ground truth values
	for (int i = 0; i < numberOfDetectedGroundTruth; i++) 
	{
		string gt;
		ground_truth >> gt;
		ground_truth.push_back(stoi(gt));
	}

	cout << "Video number: " << index << ", result: " << endl;

	auto precision = calculatePrecision(color_recordFrames, ground_truth);
	auto recall = calculateRecall(color_recordFrames, ground_truth);
	auto F1 = precision * recall / 2 (precision + recall);	
	cout << "Pixel: Precision = " << precision << ", Recal = " << recall << ", F1 = " << F1 << endl;

	precision = calculatePrecision(color_recordFrames, ground_truth);
	recall = calculateRecall(color_recordFrames, ground_truth);
	F1 = precision * recall / 2 (precision + recall);
	cout << "Histogram: Precision = " << precision << ", Recal = " << recall << ", F1 = " << F1 << endl;

	precision = calculatePrecision(color_recordFrames, ground_truth);
	recall = calculateRecall(color_recordFrames, ground_truth);
	F1 = precision * recall / 2 (precision + recall);
	cout << "Edge: Precision = " << precision << ", Recal = " << recall << ", F1 = " << F1 << endl;

	precision = calculatePrecision(color_recordFrames, ground_truth);
	recall = calculateRecall(color_recordFrames, ground_truth);
	F1 = precision * recall / 2 (precision + recall);
	cout << "Block-based: Precision = " << precision << ", Recal = " << recall << ", F1 = " << F1 << endl;

	color.close();
	hist.close();
	edge.close();
	block.close();
	ground_truth.close();

	waitKey(0);
	
	return 0;
}

int SearchMax(vector<double> arr) {

	int tmp=0;
	for (size_t i = 0; i < arr.size(); i++)
	{
		if (arr.at(i) > tmp) {
			tmp = arr.at(i);
		}
	}
	return tmp;
}

void DrawChart(vector<double> msdAry,String a) {

	int width = msdAry.size();
	int height = 400;
	Mat histImage(height, width, CV_8UC1, Scalar(0, 0, 0));
	

	for (int i = 0; i < width; i++)
	{
		line(histImage, Point(i , height),
			Point(i, height - 5*msdAry.at(i)),
			Scalar(255,0, 0), 2, 8, 0);
	}
	
	imshow("HistGram--MSD "+a, histImage);

}

Mat getLoG(const Mat& srcImage, uchar ksize, float sigma) {
	const uchar size = ksize;
	Mat kernel(ksize, ksize, CV_64F);
	uchar halfsize = ksize >> 1;

	for (int x = -halfsize; x <= halfsize; x++) {
		for (int y = -halfsize; y <= halfsize; y++)
		{
			kernel.at<double>(halfsize + x, halfsize + y) = -1 * (
				((x * x + y * y - 2 * sigma * sigma) /
					(2 * CV_PI * pow(sigma, 6))) *
				exp(-(x * x + y * y) /
					(2 * sigma * sigma)));
		}
	}

	return kernel;
}

void detectBlob(Mat& srcImage, vector<Point>& featurePts, double threhod, uchar nKernel) {
	const uchar sizeKernel = 9;
	float sigma = 0.66f;

	const int rows = srcImage.rows;
	const int cols = srcImage.cols;
	vector<Mat> vresponse;


	for (int i = 0; i < nKernel; i++) {
		//1. create laplacian of gaussian filter
		sigma = pow(k2, i);
		Mat LoG = getLoG(srcImage, sizeKernel, sigma);

		//2. convolving the image with LoG
		Mat res;
		//res = convolute(srcImage, LoG);
		filter2D(srcImage, res, srcImage.depth(), LoG);

		//stores each sigma 
		res = res.mul(res);
		vresponse.push_back(res);
		
	}
	
	//3. find the maxium peak
	for (int i = 1; i < rows - 1; ++i)
	{
		for (int j = 1; j < cols - 1; ++j)
		{
			float lmax = vresponse[0].at<uchar>(i, j);
			int r = 0;
			bool flag = true;

			//compare with it's neighbour
			for (int x = -1; x <= 1; ++x)
			{
				for (int y = -1; y <= 1; ++y)
				{
					if (vresponse[0].at<uchar>(i + x, j + y) < lmax)
					{
						flag = false;
						break;
					}
				}
				if (!flag) break;

			}

			if (!flag) break;

			//compare with larger sigma LoG filters.
			for (int k = 1; k < nKernel; k++) {

				if (vresponse[k].at<uchar>(i, j) > lmax)
				{
					lmax = vresponse[k].at<uchar>(i, j);
					r = k;
				}
			}

			//4.stores blob's coordinates to draw
			if (lmax > threhod)
			{
				featurePts.push_back(BlobFeature(Point(j, i)));
			}
		}
	}
}

float calculatePrecision(std::map<int, int> detected, vector<int> ground_truth) 
{
	auto correctFrameNumber = 0;
	auto missingFrameNumber = 0;

	for (int i = 0; i < detected.size() - 1; ++i) {
		for (int j = 0; j < ground_truth.size() - 1; ++j) {
			if (abs(detected[i].key - ground_truth[i]) < threhold_detected_frame)
				correctFrameNumber++;
			if (abs(detected[i].key - ground_truth[i]) < threhold_missing_frame)
				missingFrameNumber++;
		}
	}

	return correctFrameNumber / (correctFrameNumber + missingFrameNumber);
}

float calculateRecall(std::map<int, int> detected, vector<int> ground_truth)
{
	auto correctFrameNumber = 0;
	auto falseFrameNumber = 0;

	for (int i = 0; i < detected.size() - 1; ++i) {
		for (int j = 0; j < ground_truth.size() - 1; ++j) {
			if (abs(detected[i].key - ground_truth[i]) < threhold_detected_frame)
				correctFrameNumber++;
			if (abs(detected[i].key - ground_truth[i]) < threhold_false_frame)
				falseFrameNumber++;
		}
	}

	return correctFrameNumber / (correctFrameNumber + falseFrameNumber);
}