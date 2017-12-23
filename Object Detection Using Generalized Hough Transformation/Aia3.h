//============================================================================
// Name        : Aia3.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for second AIA assignment
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Aia3{

	public:
		// constructor
		Aia3(void){};
		// destructor
		~Aia3(void){};
		
		// processing routine
		// --> some parameters have to be set in this function
		void run(string, string);
		// testing routine
		void test(string, float, float);

	private:
		// --> these functions need to be edited
		void makeFFTObjectMask(vector<Mat>& templ, double scale, double angle, Mat& fftMask);
		vector<Mat> makeObjectTemplate(Mat& templateImage, double sigma, double templateThresh);
		vector< vector<Mat> > generalHough(Mat& gradImage, vector<Mat>& templ, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange);
		void plotHough(vector< vector<Mat> >& houghSpace);
		// given functions
		void process(Mat&, Mat&, Mat&);
		Mat makeTestImage(Mat& temp, double angle, double scale, double* scaleRange);
		Mat rotateAndScale(Mat& temp, double angle, double scale);
		Mat calcDirectionalGrad(Mat& image, double sigma);
		void showImage(Mat& img, string win, double dur);
		void circShift(Mat& in, Mat& out, int dx, int dy);
		void findHoughMaxima(vector< vector<Mat> >& houghSpace, double objThresh, vector<Scalar>& objList);
		void plotHoughDetectionResult(Mat& testImage, vector<Mat>& templ, vector<Scalar>& objList, double scaleSteps, double* scaleRange, double angleSteps, double* angleRange);
		
};
