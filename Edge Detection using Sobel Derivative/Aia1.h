//============================================================================
// Name        : Aia1.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for first AIA assignment
//============================================================================

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Aia1{

	public:
		// constructor
		Aia1(void){};
		// destructor
		~Aia1(void){};
		
		// processing routine
		void run(string);
		// testing routine
		void test(string);

	private:
		// function that performs some kind of (simple) image processing
		// --> edit ONLY this function!
		Mat doSomethingThatMyTutorIsGonnaLike(Mat&);

		// test function
		void test_doSomethingThatMyTutorIsGonnaLike(Mat&, Mat&);
};
