//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include <iostream>

#include "Aia1.h"

using namespace std;

// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

	// will contain path to input image (taken from argv[1])
	string fname;

	// check if image path was defined
	if (argc != 2){
	    cerr << "Usage: aia1 <path_to_image>" << endl;
	    return -1;
	}else{
	    // if yes, assign it to variable fname
	    fname = argv[1];
	}
	
	// construct processing object
	Aia1 aia1;

	// start processing
	aia1.run(fname);

	// run some test routines
	aia1.test(fname);

	return 0;

}
