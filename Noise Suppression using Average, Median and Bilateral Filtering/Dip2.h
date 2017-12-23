//============================================================================
// Name        : Dip2.h
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : header file for second DIP assignment
//============================================================================

#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class Dip2{

   public:
      // constructor
      Dip2(void){};
      // destructor
      ~Dip2(void){};
		
      // processing routines
      // creates noise images
      void generateNoisyImages(string);
      // for noise suppression
      void run(void);
      // testing routine
      void test(void);
	  

   private:
      // function headers of functions to be implemented
      // --> please edit ONLY these functions!
      // performs spatial convolution of image and filter kernel
      Mat spatialConvolution(Mat&, Mat&);
      // moving average filter (aka box filter)
      Mat averageFilter(Mat& src, int kSize);
      // median filter
      Mat medianFilter(Mat& src, int kSize);
      // bilateral filer
      Mat bilateralFilter(Mat& src, int kSize, double sigma);
      // non-local means filter
      Mat nlmFilter(Mat& src, int searchSize, double sigma);

      // function headers of given functions
      // performs noise reduction
      Mat noiseReduction(Mat&, string, int, double=0);

      // test functions
      void test_spatialConvolution(void);
      void test_averageFilter(void);
      void test_medianFilter(void);
};
