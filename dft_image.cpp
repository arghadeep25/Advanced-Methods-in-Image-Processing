//============================================================================
// Name        : Discrete Fourier Transform
// Author      : Arghadeep Mazumder
// Version     : 1.0
//============================================================================

#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

void takeDFT(Mat& source, Mat& destination) {
	//DFT produce complex image i.e real and imaginary. So we need a matrix having 2 channel
	Mat originalcomplex[2] = { source, Mat::zeros(source.size(),source.type()) };
	Mat dftready, dftoriginal;
	//We need to merge both the images in to a single image
	merge(originalcomplex, 2, dftready);//input complex & output dft ready
	dft(dftready, dftoriginal, DFT_COMPLEX_OUTPUT); //For both real and imaginary
	destination = dftoriginal; //passing the value to the destination
}

void recenterdft(Mat& source) {
	//Recentering the DFT i.e the low frequency information in the center and the high frequency at the corner
	//     Without Recentering         After Recentering
	//       ________________           ________________
	//       |      |       |           |      |       |
	//       |   q1 |   q2  |           |   q4 |   q3  |
	//       |______|_______|           |______|_______|
	//       |      |       |    ====>  |      |       |
	//       |  q3  |   q4  |           |  q2  |   q1  |
	//       |      |       |           |      |       |
	//       ----------------           ----------------
	// If we do not recenter then the lower frequency will be at the corners
	int centerX = source.cols / 2;
	int centerY = source.rows / 2;
	Mat q1(source, Rect(0, 0, centerX, centerY));
	Mat q2(source, Rect(centerX, 0, centerX, centerY));
	Mat q3(source, Rect(0, centerY, centerX, centerY));
	Mat q4(source, Rect(centerX, centerY, centerX, centerY));

	Mat swapmap;
	q1.copyTo(swapmap);
	q4.copyTo(q1);
	swapmap.copyTo(q4);

	q1.copyTo(swapmap);
	q4.copyTo(q1);
	swapmap.copyTo(q4);

}

void showdft(Mat& source) {
	//2 channel corresponds to the real and imaginary part 
	Mat splitarray[2] = { Mat::zeros(source.size(),CV_32F),Mat::zeros(source.size(),CV_32F) };
	//Splitting the Source image to real and imaginary and paasing to the 2 channel Mat
	split(source, splitarray); 
	Mat dftmag;
	//As it composed of real and imaginary part so we need to take the magnitude and paasing to dftmag matrix
	//Mag= sqrt((Re)^2+(Im)^2) 
	//[0] for real and [1] for imaginary
	magnitude(splitarray[0], splitarray[1], dftmag); 
	//As the values of the magnitude are in the enormous range we need to take the logarithmic value
	//The values of the matrix are not known. So adding 1 to all the elements of the matrix [log(1)=0]
	dftmag += Scalar::all(1);  
	log(dftmag, dftmag); //log(input,output)
	//As the values are still beyond 0 and 1 so we need to normalize the value to lie between the range
	normalize(dftmag, dftmag,0,1,CV_MINMAX);  //normalize(input, output, low_range,max_range)
	//recenterdft(dftmag);
	imshow("DFT", dftmag);
	waitKey();
}
int main(int argc, char** argv) {
	Mat original = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE); //Loading Grayscale Image (Important for DFT)
	//DFT for color image needs to repeat the whole process 3 times as color images are of 3 channel
	//Also we need to split all the channels into grey scale
	Mat originalfloat, dftoriginal;
	//Original image lies between 0 and 255 as it is grey scale CV_8U.
	original.convertTo(originalfloat, CV_32FC1, 1.0 / 255.0); //For single chanenl
	//Normalizing the value so that they range between 0 and 1
	takeDFT(originalfloat, dftoriginal);
	showdft(dftoriginal);
	return 0;
}
