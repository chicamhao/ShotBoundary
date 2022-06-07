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

#include "rapidxml.hpp"
#include "rapidxml_print.hpp"


using namespace std;
using namespace cv;
using namespace rapidxml;

//Initializing Global Variable for xml ->  Logo Transition
xml_document<> docLogo;
//Initializing Global Variable for xml ->  Cut Transition
xml_document<> docCut;

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";



int threshold_type = 1;

////   -------    640 X 360
///// -> LOGO TRANSITION PARAMETERS
int threshold_value_msd = 26;
int const max_value_msd = 60;

 //////- > CUT TRANSITION PARAMETERS
int const max_value = 100;
int const max_type = 4;
int threshold_value = 16;
std::map<int, int> color_recordFrames;
std::map<int, int> histogram_recordFrames;
std::map<int, int> edge_recordFrames;

int const max_BINARY_value = 255;
bool DebugMode = false;

//arrays
double** imd; // store MSD array..
// vectors
vector<double> msdVector1, msdVector2, msdVector3;

//tempory for right gradient
vector<double> rightGradientVector;
// Methods
void DrawHist11(int, int, Mat, int);
void DrawChart(vector<double>,String);
int SearchMax(vector<double>);
double tmpMsd=0;

//histogram parameter
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
Mat frameEdge, frameEdge2;
Mat delate, delate2;
Mat invert, invert2;
Mat element = getStructuringElement(MORPH_RECT,
	Size(2 * dilation_size + 1, 2 * dilation_size + 1),
	Point(dilation_size, dilation_size));

void runProgramX(String path) 
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

		// 
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

		//-----------------------------------------------------------------------------------------
		//
		//-----------------------EDGE CHANGE RATIO-------------------------------------------------
		//vector<Point> myedges1;
		//vector<Point> myedges2;

		//Canny(frameGray1, frameEdge, 50, 150, 3, false);
		//Canny(frameGray2, frameEdge2, 50, 150, 3, false);

		//dilate(frameEdge, delate, element);
		//dilate(frameEdge2, delate2, element);

		//auto invert = (255 - delate);
		//auto invert2 = (255 - delate2);

		//auto and1 = (frameEdge & invert);
		//auto and2 = (frameEdge2 & invert2);

		//auto pixel_sum_1 = sum(frameEdge)[0];
		//auto pixel_sum_2 = sum(frameEdge2)[0];

		//auto out_pixels = sum(and1)[0];
		//auto in_pixels = sum(and2)[0];

		//auto edgeWeight = max(in_pixels / pixel_sum_2, out_pixels / pixel_sum_1);
		//cout << "EDGE CHANGE RATIO: Weight: " << edgeWeight << ", frame: " << frameNo << endl;
		//edge_recordFrames.insert(std::pair<int, int>(frameNo, edgeWeight));
		//msdVector3.push_back(edgeWeight);

		//int milliseconds = cap.get(CAP_PROP_POS_MSEC);
		//int timeUtillXML = (int) milliseconds / 1000;

		//int seconds = (int)(milliseconds / 1000) % 60;
		//int minutes = (int)((milliseconds / (1000 * 60)) % 60);
		//int hours = (int)((milliseconds / (1000 * 60 * 60)) % 24);

		//DrawChart(msdVector1,"1");
		//DrawChart(msdVector2, "2");
		//DrawChart(msdVector3, "3");

		//imshow("18120353", frameColor); //show the frame in "MyVideo" window
	
		//if (waitKey(50) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		//{
		//	cout << "esc key is pressed by user" << endl;
		//	break;
		//}
	}

	// When everything done, release the video capture object
	cap.release();

	// Declare vector of pairs
	vector<pair<int, int> > A;
	vector<pair<int, int> > B;
	vector<pair<int, int> > C;

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

	// Sort using comparator function
	sort(A.begin(), A.end(), [](const pair<int, int>& left, const pair<int, int>& right) {return left.second > right.second; });
	sort(B.begin(), B.end(), [](const pair<int, int>& left, const pair<int, int>& right) {return left.second > right.second; });
	sort(C.begin(), C.end(), [](const pair<int, int>& left, const pair<int, int>& right) {return left.second > right.second; });

	// Print the sorted value
	int count = 0;
	ofstream color, hist, edge;
	color.open("output/color_output.txt");
	hist.open("output/histogram_output.txt");
	edge.open("output/edge_output.txt");

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

	color.close();
	hist.close();
	edge.close();

	waitKey(0);
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
