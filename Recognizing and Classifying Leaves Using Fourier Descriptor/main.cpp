//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include <iostream>

#include "Aia2.h"

using namespace std;

// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

	// will contain path to input image (taken from argv[1])
	string img, tmpl1, tmpl2;

	// check if image path was defined
	// check if image paths were defined
	if (argc != 4){
	    cerr << "Usage: aia2 <input image>  <class 1 example>  <class 2 example>" << endl;
	    cerr << "Press enter to continue..." << endl;
	    cin.get();
	    return -1;
	}else{
	    // if yes, assign it to variable fname
	    img = argv[1];
	    tmpl1 = argv[2];
	    tmpl2 = argv[3];
	}
	
	// construct processing object
	Aia2 aia2;

	// run some test routines
	aia2.test();

	// start processing
	aia2.run(img, tmpl1, tmpl2);

	return 0;

}
