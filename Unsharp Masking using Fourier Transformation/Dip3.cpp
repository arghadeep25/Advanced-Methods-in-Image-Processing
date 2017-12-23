//============================================================================
// Name    : Dip3.cpp
// Author   : Ronny Haensch
// Version    : 2.0
// Copyright   : -
// Description : Exercise 3 [Unsharp Masking]
// Edited by   : Arghadeep Mazumder
//				 Michael Tamakloe
//============================================================================

#include "Dip3.h"

// Generates a gaussian filter kernel of given size
/*
kSize:     kernel size (used to calculate standard deviation)
return:    the generated filter kernel
*/
Mat Dip3::createGaussianKernel(int kSize){

	//Defining Parameters
	double sigma_x = kSize / 5;
	double sigma_y = kSize / 5;
	double u_x = (kSize) / 2;
	double u_y = (kSize) / 2;
	double x_part, y_part, normalized_kernel;
	//Defining Kernel Matrix
	Mat gauss_kernel = Mat::zeros(kSize, kSize, CV_32FC1);
	//Defining Gaussian Kernel
	for (int ker_row = 0;ker_row < kSize;ker_row++) {
		for (int ker_col = 0;ker_col < kSize;ker_col++) {
			gauss_kernel.at<float>(ker_row, ker_col) = (1 / (2 * CV_PI*sigma_x*sigma_y))*exp(-0.5*(pow(((ker_row - u_x) / sigma_x), 2) + pow(((ker_col - u_y) / sigma_y), 2)));
		}
	}
   //Normalizing the Kernel
	normalized_kernel = sum(gauss_kernel)[0];
	gauss_kernel /= normalized_kernel;
	return gauss_kernel;
}


// Performes a circular shift in (dx,dy) direction
/*
in       input matrix
dx       shift in x-direction
dy       shift in y-direction
return   circular shifted matrix
*/
Mat Dip3::circShift(Mat& in, int dx, int dy){

	//Creating a new Output Matrix
	Mat output = Mat::zeros(in.size(), CV_32FC1);
	//Performing Circular Shift
	for (int image_rows = 0;image_rows < in.rows;image_rows++) {
		for (int image_cols = 0;image_cols < in.cols;image_cols++) {
			int new_rows = (image_rows + dx+in.rows) % in.rows;
			int new_cols = (image_cols + dy+in.cols) % in.cols;
			output.at<float>(new_rows, new_cols) = in.at<float>(image_rows,image_cols);
		}
   }
   return output;
}

//Performes a convolution by multiplication in frequency domain
/*
in       input image
kernel   filter kernel
return   output image
*/
Mat Dip3::frequencyConvolution(Mat& in, Mat& kernel){

	//Declaring new matrix of image size where the DFT output will be copied
	Mat dft_output = Mat::zeros(in.size(),in.type());
	//Declaring new matrix of image size where the kernel matrix will be copied
	Mat new_kernel = Mat::zeros(in.size(), in.type());
	//copying the kernel matrix to the new matrix
	kernel.copyTo(new_kernel( Rect(0, 0, kernel.rows, kernel.cols)));
	// Performing circular shift
	new_kernel = circShift(new_kernel, -1, -1);
	//Fourier Transform of Image
	dft(in, dft_output, 0);
	//Fourier Transform of Kernel
	dft(new_kernel, new_kernel, 0);
	//Performing per element multiplication of Two fourier transform
	mulSpectrums(dft_output, new_kernel, new_kernel, 0);
	//Inversing the Transform and normalizing the values
	dft(new_kernel, dft_output, DFT_INVERSE + DFT_SCALE);
	return dft_output;
}

// Performs UnSharp Masking to enhance fine image structures
/*
in       the input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain; 2 <==> seperable filter; 3 <==> integral image
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::usm(Mat& in, int type, int size, double thresh, double scale){

   // some temporary images 
   Mat tmp(in.rows, in.cols, CV_32FC1);
   // Creating a new matrix where output will be copied
   Mat temp = Mat::zeros(in.size(),in.type());
   // calculate edge enhancement

   // 1: smooth original image
   //    save result in tmp for subsequent usage
   switch(type){
      case 0:
         tmp = mySmooth(in, size, 0);
         break;
      case 1:
         tmp = mySmooth(in, size, 1);
         break;
      case 2: 
	tmp = mySmooth(in, size, 2);
        break;
      case 3: 
	tmp = mySmooth(in, size, 3);
        break;
      default:
         GaussianBlur(in, tmp, Size(floor(size/2)*2+1, floor(size/2)*2+1), size/5., size/5.);
   }
   // I2=I0-I1 (Original - Smooth)
   tmp = in - tmp;
   //Performing addition 
   for (int x = 0; x < in.rows; x++) for (int y = 0; y < in.cols; y++) {
	   if (tmp.at<float>(x, y) > thresh) { // Checking the value larger than threshold
		   //I4=I0+(s*I2)
		   temp.at<float>(x, y) = in.at<float>(x, y) + tmp.at<float>(x, y) * scale;
	   }
   }
   return temp;

}

// convolution in spatial domain
/*
src:    input image
kernel:  filter kernel
return:  convolution result
*/
Mat Dip3::spatialConvolution(Mat& src, Mat& kernel){

	float convolution_op;
	//Defining the border
	int top_border = (int)(0.5*kernel.rows); int bottom_border = (int)(0.5*kernel.rows);
	int left_border = (int)(0.5*kernel.cols); int right_border = (int)(0.5*kernel.cols);
	Mat output_image = Mat::zeros(src.rows, src.cols, CV_32FC1);
	//For Border handling extending the input image with border
	Mat extended_input = Mat::zeros(src.rows + kernel.rows, src.cols + kernel.cols, CV_32FC1);
	//Function copies the source image into the middle of the destination image
	copyMakeBorder(src, extended_input, top_border, bottom_border, left_border, right_border, BORDER_CONSTANT);
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
			output_image.at<float>(image_row, image_col) = convolution_op / sum(kernel)[0]; //Normalization of the Result Image
																							//If we dont normalize the resulting image it will give abstract output
		}
	}
	return output_image;
}

// convolution in spatial domain by seperable filters
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::seperableFilter(Mat& src, int size){

   // optional

   return src;

}

// convolution in spatial domain by integral images
/*
src:    input image
size     size of filter kernel
return:  convolution result
*/
Mat Dip3::satFilter(Mat& src, int size){

   // optional

   return src;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function calls processing function
/*
in       input image
type     integer defining how convolution for smoothing operation is done
         0 <==> spatial domain; 1 <==> frequency domain
size     size of used smoothing kernel
thresh   minimal intensity difference to perform operation
scale    scaling of edge enhancement
return   enhanced image
*/
Mat Dip3::run(Mat& in, int smoothType, int size, double thresh, double scale){

   return usm(in, smoothType, size, thresh, scale);

}


// Performes smoothing operation by convolution
/*
in       input image
size     size of filter kernel
type     how is smoothing performed?
return   smoothed image
*/
Mat Dip3::mySmooth(Mat& in, int size, int type){

   // create filter kernel
   Mat kernel = createGaussianKernel(size);
 
   // perform convoltion
   switch(type){
     case 0: return spatialConvolution(in, kernel);	// 2D spatial convolution
     case 1: return frequencyConvolution(in, kernel);	// 2D convolution via multiplication in frequency domain
     case 2: return seperableFilter(in, size);	// seperable filter
     case 3: return satFilter(in, size);		// integral image
     default: return frequencyConvolution(in, kernel);
   }
}

// function calls basic testing routines to test individual functions for correctness
void Dip3::test(void){

   test_createGaussianKernel();
   test_circShift();
   test_frequencyConvolution();
   cout << "Press enter to continue"  << endl;
   cin.get();

}

void Dip3::test_createGaussianKernel(void){

   Mat k = createGaussianKernel(11);
   
   if ( abs(sum(k).val[0] - 1) > 0.0001){
      cout << "ERROR: Dip3::createGaussianKernel(): Sum of all kernel elements is not one!" << endl;
      return;
   }
   if (sum(k >= k.at<float>(5,5)).val[0]/255 != 1){
      cout << "ERROR: Dip3::createGaussianKernel(): Seems like kernel is not centered!" << endl;
      return;
   }
   cout << "Message: Dip3::createGaussianKernel() seems to be correct" << endl;
}

void Dip3::test_circShift(void){
   
   Mat in = Mat::zeros(3,3,CV_32FC1);
   in.at<float>(0,0) = 1;
   in.at<float>(0,1) = 2;
   in.at<float>(1,0) = 3;
   in.at<float>(1,1) = 4;
   Mat ref = Mat::zeros(3,3,CV_32FC1);
   ref.at<float>(0,0) = 4;
   ref.at<float>(0,2) = 3;
   ref.at<float>(2,0) = 2;
   ref.at<float>(2,2) = 1;
   
   if (sum((circShift(in, -1, -1) == ref)).val[0]/255 != 9){
      cout << "ERROR: Dip3::circShift(): Result of circshift seems to be wrong!" << endl;
      return;
   }
   cout << "Message: Dip3::circShift() seems to be correct" << endl;
}

void Dip3::test_frequencyConvolution(void){
   
   Mat input = Mat::ones(9,9, CV_32FC1);
   input.at<float>(4,4) = 255;
   Mat kernel = Mat(3,3, CV_32FC1, 1./9.);

   Mat output = frequencyConvolution(input, kernel);
   
   if ( (sum(output < 0).val[0] > 0) || (sum(output > 255).val[0] > 0) ){
      cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains too large/small values!" << endl;
      return;
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
            cout << "ERROR: Dip3::frequencyConvolution(): Convolution result contains wrong values!" << endl;
            return;
         }
      }
   }
   cout << "Message: Dip3::frequencyConvolution() seems to be correct" << endl;
}
