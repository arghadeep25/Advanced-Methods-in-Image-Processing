//============================================================================
// Name        : Aia4.h
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : header file for fourth AIA assignment
//============================================================================

#include <iostream>
#include <stdio.h>
#include <limits>

#include <list>
#include <vector>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct comp {Mat mean; Mat covar; double weight;};

class Aia5{

	public:
		// constructor
		Aia5(void){};
		// destructor
		~Aia5(void){};
		
		// processing routine
		void run(void);
		void test(void);
		
	private:
		// functions to be written
		Mat calcCompLogL(vector<struct comp*>& model, Mat& features);
		Mat calcMixtureLogL(vector<struct comp*>& model, Mat& features);
		Mat gmmEStep(vector<struct comp*>& model, Mat& features);
		void gmmMStep(vector<struct comp*>& model, Mat& features, Mat& posterior);

		// given functions
		void initNewComponent(vector<struct comp*>& model, Mat& features);
		void plotGMM(vector<struct comp*>& model, Mat& features);
		void trainGMM(Mat& data, int numberOfComponents, vector<struct comp*>& model);
		void readImageDatabase(string dataPath, vector<Mat>& db);
		void genFeatureProjection(vector<Mat>& imgDB, vector<PCA>& featSpace, int vectLen);

};
