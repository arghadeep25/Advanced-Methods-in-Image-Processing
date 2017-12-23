//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing routines
//============================================================================

#include <iostream>

#include "Aia4.h"

using namespace std;

/* usage:
  performs EM to estimate parameters of a Gaussian mixture model
*/
// main function
int main(int argc, char** argv) {

	// construct processing object
	Aia5 aia;

	// start simple clustering
   aia.test();
	
	// start processing
   aia.run();
	
	cout << endl << "Continue with pressing enter..." << endl;
	cin.get();
	
	return 0;

}
