//============================================================================
// Name        : Aia2.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for second AIA assignment
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Aia2{

	public:
		// constructor
		Aia2(void){};
		// destructor
		~Aia2(void){};
		
		// processing routine
		// --> some parameters have to be set in this function
		void run(string, string, string);
		// testing routine
		void test(void);

	private:
		// --> these functions need to be edited
		void getContourLine(const Mat& contourImage, vector<Mat>& objList, int thresh, int k);
		Mat makeFD(const Mat& contour);
		Mat normFD(const Mat& fd, int n);
		void plotFD(const Mat& fd, string win, double dur=-1);
		
		// given functions
		void showImage(const Mat& img, string win, double dur=-1);

		// test function
		void test_getContourLine(void);
		void test_makeFD(void);
		void test_normFD(void);
		
};
