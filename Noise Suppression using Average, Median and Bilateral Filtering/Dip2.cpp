//============================================================================
// Name        : Dip2.cpp
// Author      : Ronny Haensch
// Version     : 2.0
// Copyright   : -
// Description : Exercise 2 [Noise Suppression]
// Edited by   : Arghadeep Mazumder
//				 Michael Tmakloe
//============================================================================

#include "Dip2.h"

// convolution in spatial domain
/*
src:     input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip2::spatialConvolution(Mat& src, Mat& kernel) {
	
	float convolution_op;
	//Defining the border
	int top_border = (int)(0.5*kernel.rows); int bottom_border = (int)(0.5*kernel.rows);
	int left_border = (int)(0.5*kernel.cols); int right_border = (int)(0.5*kernel.cols);
	Mat output_image = Mat::zeros(src.rows, src.cols, CV_32FC1);
	//For Border handling extending the input image with border
	Mat extended_input = Mat::zeros(src.rows + kernel.rows, src.cols +  kernel.cols, CV_32FC1);
	//Function copies the source image into the middle of the destination image
	copyMakeBorder(src, extended_input, top_border,bottom_border,left_border,right_border, BORDER_CONSTANT);
	//Flipping the Kernel
	flip(kernel, kernel, -1); //***Tried flipping the kernel inside without using the flip funciton
	//Error: "Unhandled exception at 0x00007FFE1BE38B9C in Exe_2.exe: Microsoft C++ exception: cv::Exception at memory location 0x000000417ACAC4F0."
	//Convolution of Kernel and the Source Image
	
	//Element in the Matrix denoted by Image_ij
	for (int image_row = 0;image_row < src.rows;image_row++) { //Selecting the row of the Image Matrix
		for (int image_col = 0;image_col < src.cols;image_col++) { //Selecting the column of the Image Matrix
			convolution_op = 0; //Setting the value to zero after execution of the for loop
			for (int kernel_row = 0;kernel_row < kernel.rows;kernel_row++) { //Selecting the row of the Kernel Matrix
				for (int kernel_col = 0;kernel_col < kernel.cols;kernel_col++) { //Selecting the column of the Kernel Matrix
					int new_row = kernel_row + image_row;
					int new_col = kernel_col + image_col;
					//Convolution is defined as Sum[f(x,y)*h{(x-alpha),(y-beta)}]
					//Here extended_input is f(x,y) and kernel is h{(x-alpha),(y-beta)}
					convolution_op += extended_input.at<float>(new_row, new_col)*kernel.at<float>(kernel_row, kernel_col);
				}
			}
			output_image.at<float>(image_row, image_col) = convolution_op/sum(kernel)[0]; //Normalization of the Result Image
			//If we dont normalize the resulting image it will give abstract output
		}
	}
	return output_image;
}

// the average filter
// HINT: you might want to use Dip2::spatialConvolution(...) within this function
/*
src:     input image
kSize:   window size used by local average
return:  filtered image
*/
/*Using Average Filtering for Gaussian Noise*/
Mat Dip2::averageFilter(Mat& src, int kSize){
	//Kernel
	Mat kernel = Mat::ones(kSize, kSize, CV_32FC1);
	//Calling spatialConvolution for Average Filtering
	Mat output_image = spatialConvolution(src, kernel);
   return output_image;
}

// the median filter
/*
src:     input image
kSize:   window size used by median operation
return:  filtered image
*/
Mat Dip2::medianFilter(Mat& src, int kSize){
	
	//Defining the border 
	int top_border = (int)(0.5*kSize); int bottom_border = (int)(0.5*kSize);
	int left_border = (int)(0.5*kSize); int right_border = (int)(0.5*kSize);
	Mat result_image = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat extended_input = Mat::zeros(src.rows + kSize , src.cols + kSize, CV_32FC1);
	//For Border handling extending the input image with border
	copyMakeBorder(src, extended_input, top_border, bottom_border, left_border, right_border, BORDER_CONSTANT);
	int array_size = pow(kSize, 2);
	//Better to use vector than array as for defining array we need constant value but here the size of kernel is flexible
	vector<float> kernel_array = {};
	for (int image_row = 0;image_row < src.rows;image_row++) { //Selecting the row of the Image Matrix
		for (int image_col = 0;image_col < src.cols;image_col++) { //Selecting the column of the Image Matrix
			kernel_array.clear(); //Clearing the array after the execution of the for loop 
			for (int kernel_row = 0;kernel_row < kSize;kernel_row++) { //Selecting the row of the Kernel Matrix
				for (int kernel_col = 0;kernel_col < kSize;kernel_col++) { //Selecting the column of the Kernel Matrix
					// Local Window of size kSize*kSize has been selected
					//Inserting the intensities from the Extended Image to the Local Window
					kernel_array.push_back(extended_input.at<float>(kernel_row + image_row, kernel_col + image_col));
				}
			}
			//Soring the Window Elements based on intensities
			sort(kernel_array.begin(), kernel_array.end());
			//Taking the local median and passing it to the resulting image one by one
			result_image.at<float>(image_row, image_col) = kernel_array[(array_size) / 2]; 
		}
	}
		
   return result_image;
}

// the bilateral filter
/*
src:     input image
kSize:   size of the kernel --> used to compute std-dev of spatial kernel
sigma:   standard-deviation of the radiometric kernel
return:  filtered image
*/
Mat Dip2::bilateralFilter(Mat& src, int kSize, double sigma){
	int size_w = (kSize - 1) / 2;
	Mat output_image = Mat::zeros(src.rows, src.cols, CV_32FC1);
	//Spatial Kernel
	Mat spatial_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
	//Radiometric Kernel
	Mat radiometric_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
	//Combined
	Mat combined_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
	//A temporary kernel of kSize has been taken
	Mat temp_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
		for (int image_row = 0;image_row < src.rows;image_row++) { //Selecting the row of the Image Matrix
			for (int image_col = 0;image_col < src.cols;image_col++) { //Selecting the column of the Image Matrix
			//Clearing all the Kernels
			combined_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
			spatial_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
			radiometric_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
			temp_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
			for (int kernel_rows = 0;kernel_rows < temp_kernel.rows;kernel_rows++) { //Selecting the row of the Kernel Matrix
				for (int kernel_cols = 0;kernel_cols < temp_kernel.cols;kernel_cols++) { //Selecting the column of the Kernel Matrix
					int rows = image_row - kernel_rows + size_w;
					int cols = image_col - kernel_cols + size_w;
					//Checking Border
					if (rows<0 || cols<0 || rows>=src.rows || cols>=src.cols) { 
						temp_kernel.at<float>(kernel_rows, kernel_cols) = 0;
					}
					else {
						//Passing the intensity values from Image Matrix to Kernel Matrix
						temp_kernel.at<float>(kernel_rows, kernel_cols) = src.at<float>(rows, cols);
					}
					//Spatial Weight
					double spatial = 1 / (2 * CV_PI*pow(kSize, 2))*exp(-(pow(kernel_rows - size_w, 2) + pow(kernel_cols - size_w, 2)) / (2 * pow(kSize, 2)));
					//Radiometric  Kernel
					double radiometric = 1 / (2 * CV_PI*pow(sigma, 2))*exp(-(pow(temp_kernel.at<float>(kernel_rows, kernel_cols) - src.at<float>(kernel_rows, kernel_cols), 2)) / (2.0 * pow(sigma, 2)));
					//Spatial Kernel
					spatial_kernel.at<float>(kernel_rows, kernel_cols) = spatial;
					//Radiometric Kenrel
					radiometric_kernel.at<float>(kernel_rows, kernel_cols) = radiometric;
				}
			}
			//Taking the scalar value for Normalization
			double spatial_sum = sum(spatial_kernel)[0]; //Scalar Sum
			double radiometric_sum = sum(radiometric_kernel)[0]; //Scalar Sum
			for (int temp_pass_val_rows = 0;temp_pass_val_rows < kSize;temp_pass_val_rows++) {
				for (int temp_pass_val_col = 0;temp_pass_val_col < kSize;temp_pass_val_col++) {
					//Normalizing Spatial Kernel
					spatial_kernel.at<float>(temp_pass_val_rows, temp_pass_val_col) /= spatial_sum; 
					//Normalizing Radiomentric Kernel
					radiometric_kernel.at<float>(temp_pass_val_rows, temp_pass_val_col) /= radiometric_sum;
				}
			}
			//Multiplying the elements of two matrices elementwise
			combined_kernel = spatial_kernel.mul(radiometric_kernel);
			double sum_combined_kernel = sum(combined_kernel)[0]; //Scalar Value
			for (int final_pass_row = 0;final_pass_row < combined_kernel.rows;final_pass_row++) { 
				for (int final_pass_col = 0;final_pass_col < combined_kernel.cols;final_pass_col++) {
					combined_kernel.at<float>(final_pass_row, final_pass_col) /= sum_combined_kernel;
				}
			}
			output_image.at<float>(image_row, image_col) = sum(temp_kernel.mul(combined_kernel))[0];
		}
	}
    return output_image;

}

// the non-local means filter
/*
src:   		input image
searchSize: size of search region
sigma: 		Optional parameter for weighting function
return:  	filtered image
*/
Mat Dip2::nlmFilter(Mat& src, int searchSize, double sigma){
  
    return src.clone();

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing function, and saves result
void Dip2::run(void){

   // load images as grayscale
	cout << "load images" << endl;
	Mat noise1 = imread("noiseType_1.jpg", 0);
   if (!noise1.data){
	   cerr << "noiseType_1.jpg not found" << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
	   exit(-3);
	}
  
  noise1.convertTo(noise1, CV_32FC1);
  
	
   Mat noise2 = imread("noiseType_2.jpg",0);
	if (!noise2.data){
	   cerr << "noiseType_2.jpg not found" << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
	   exit(-3);
	}
   noise2.convertTo(noise2, CV_32FC1);
	cout << "done" << endl;
	  
   // apply noise reduction
	// TO DO !!!
	// ==> Choose appropriate noise reduction technique with appropriate parameters
	// ==> "average" or "median"? Why?
	// ==> try also "bilateral" (and if implemented "nlm")
	cout << "reduce noise" << endl;
	Mat restorated1 = noiseReduction(noise1, "average", 5); //Change the value of kSize for Kernel Size [** Odd Numbers only]
	Mat restorated2 = noiseReduction(noise2, "average",5);
	/*string win1 = string("Noise Image 1");
	namedWindow(win1.c_str());
	imshow(win1.c_str(), restorated1);*/
	cout << "done" << endl;
	  
	// save images
	cout << "save results" << endl;
	imwrite("restorated1.jpg", restorated1);
	imwrite("restorated2.jpg", restorated2);
	cout << "done" << endl;

}

// noise reduction
/*
src:     input image
method:  name of noise reduction method that shall be performed
	     "average" ==> moving average
         "median" ==> median filter
         "bilateral" ==> bilateral filter
         "nlm" ==> non-local means filter
kSize:   (spatial) kernel size
param:   if method == "bilateral", standard-deviation of radiometric kernel; if method == "nlm", (optional) parameter for similarity function
         can be ignored otherwise (default value = 0)
return:  output image
*/
Mat Dip2::noiseReduction(Mat& src, string method, int kSize, double param){

	// apply moving average filter
   if (method.compare("average") == 0){
      return averageFilter(src, kSize);
   }
   // apply median filter
   if (method.compare("median") == 0){
      return medianFilter(src, kSize);
   }
   // apply bilateral filter
   if (method.compare("bilateral") == 0){
      return bilateralFilter(src, kSize, param);
   }
   // apply adaptive average filter
   if (method.compare("nlm") == 0){
      return nlmFilter(src, kSize, param);
   }

   // if none of above, throw warning and return copy of original
   cout << "WARNING: Unknown filtering method! Returning original" << endl;
   cout << "Press enter to continue"  << endl;
   cin.get();
   return src.clone();

}

// generates and saves different noisy versions of input image
/*
fname:   path to the input image
*/
void Dip2::generateNoisyImages(string fname){
 
   // load image, force gray-scale
   cout << "load original image" << endl;
   Mat img = imread(fname,0);
   if (!img.data){
      cerr << "ERROR: file " << fname << " not found" << endl;
      cout << "Press enter to exit"  << endl;
      cin.get();
      exit(-1);
   }
   string win1 = string("Original image");
   namedWindow(win1.c_str());
   imshow(win1.c_str(), img);
   // convert to floating point precision
   img.convertTo(img,CV_32FC1);
   cout << "done" << endl;
   
   // save original
   //imwrite("original.jpg", img);
	  
   // generate images with different types of noise
   cout << "generate noisy images" << endl;

   // some temporary images
   Mat tmp1(img.rows, img.cols, CV_32FC1);
   Mat tmp2(img.rows, img.cols, CV_32FC1);
   // first noise operation
   float noiseLevel = 0.15;
   randu(tmp1, 0, 1);
   threshold(tmp1, tmp2, noiseLevel, 1, CV_THRESH_BINARY);
   multiply(tmp2,img,tmp2);
   threshold(tmp1, tmp1, 1-noiseLevel, 1, CV_THRESH_BINARY);
   tmp1 *= 255;
   tmp1 = tmp2 + tmp1;
   threshold(tmp1, tmp1, 255, 255, CV_THRESH_TRUNC);

   namedWindow("Noise Image 1", WINDOW_AUTOSIZE);
   imshow("Noise Image 1", tmp1);  //Shot Noise
   // save image
   imwrite("noiseType_1.jpg", tmp1);
    
   // second noise operation
   noiseLevel = 50;
   randn(tmp1, 0, noiseLevel); //Gaussian 
   tmp1 = img + tmp1;
   threshold(tmp1,tmp1,255,255,CV_THRESH_TRUNC);
   threshold(tmp1,tmp1,0,0,CV_THRESH_TOZERO);
   // save image
   imwrite("noiseType_2.jpg", tmp1);

	cout << "done" << endl;
	cout << "Please run now: dip2 restorate" << endl;

}

// function calls some basic testing routines to test individual functions for correctness
void Dip2::test(void){

	test_spatialConvolution();
   //test_averageFilter();
   //test_medianFilter();

   cout << "Press enter to continue"  << endl;
   cin.get();

}

// checks basic properties of the convolution result
void Dip2::test_spatialConvolution(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = spatialConvolution(input, kernel);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::spatialConvolution(): input.size != output.size --> Wrong border handling?" << endl;
      return;
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::spatialConvolution(): Border of convolution result contains too large/small values --> Wrong border handling?" << endl;
         return;
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains too large/small values!" << endl;
            return;
      }
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip2::spatialConvolution(): Convolution result contains wrong values!" << endl;
            return;
         }
      }
   }
   input.setTo(0);
   input.at<float>(4,4) = 255;
   kernel.setTo(0);
   kernel.at<float>(0,0) = -1;
   output = spatialConvolution(input, kernel);
   if ( abs(output.at<float>(5,5) + 255.) < 0.0001 ){
      cout << "ERROR: Dip2::spatialConvolution(): Is filter kernel \"flipped\" during convolution? (Check lecture/exercise slides)" << endl;
      return;
   }
   if ( ( abs(output.at<float>(2,2) + 255.) < 0.0001 ) || ( abs(output.at<float>(4,4) + 255.) < 0.0001 ) ){
      cout << "ERROR: Dip2::spatialConvolution(): Is anchor point of convolution the centre of the filter kernel? (Check lecture/exercise slides)" << endl;
      return;
   }
   cout << "Message: Dip2::spatialConvolution() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void Dip2::test_averageFilter(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;

   Mat output = averageFilter(input, 3);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::averageFilter(): input.size != output.size --> Wrong border handling?" << endl;
      return;
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::averageFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
         return;
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::averageFilter(): Result contains too large/small values!" << endl;
            return;
      }
   }
   float ref[9][9] = {{0, 0, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, (8+255)/9., (8+255)/9., (8+255)/9., 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 1, 1, 1, 1, 1, 1, 1, 0},
                      {0, 0, 0, 0, 0, 0, 0, 0, 0}};
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - ref[y][x]) > 0.0001){
            cout << "ERROR: Dip2::averageFilter(): Result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip2::averageFilter() seems to be correct" << endl;
}

// checks basic properties of the filtering result
void Dip2::test_medianFilter(void){

   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;

   Mat output = medianFilter(input, 3);
   
   if ( (input.cols != output.cols) || (input.rows != output.rows) ){
      cout << "ERROR: Dip2::medianFilter(): input.size != output.size --> Wrong border handling?" << endl;
      return;
   }
  if ( (sum(output.row(0) < 0).val[0] > 0) ||
           (sum(output.row(0) > 255).val[0] > 0) ||
           (sum(output.row(8) < 0).val[0] > 0) ||
           (sum(output.row(8) > 255).val[0] > 0) ||
           (sum(output.col(0) < 0).val[0] > 0) ||
           (sum(output.col(0) > 255).val[0] > 0) ||
           (sum(output.col(8) < 0).val[0] > 0) ||
           (sum(output.col(8) > 255).val[0] > 0) ){
         cout << "ERROR: Dip2::medianFilter(): Border of result contains too large/small values --> Wrong border handling?" << endl;
         return;
   }else{
      if ( (sum(output < 0).val[0] > 0) ||
         (sum(output > 255).val[0] > 0) ){
            cout << "ERROR: Dip2::medianFilter(): Result contains too large/small values!" << endl;
            return;
      }
   }
   for(int y=1; y<8; y++){
      for(int x=1; x<8; x++){
         if (abs(output.at<float>(y,x) - 1.) > 0.0001){
            cout << "ERROR: Dip2::medianFilter(): Result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip2::medianFilter() seems to be correct" << endl;

}

