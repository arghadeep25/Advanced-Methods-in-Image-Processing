//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include <iostream>

#include "Aia3.h"

using namespace std;

/* usage:
  first case (testing): aia3 <path to template>
  second case (application): aia3 <path to template> <path to testimage>
*/
// main function
int main(int argc, char** argv) {

	// check if image paths were defined
    if ( (argc != 2) and (argc != 3) ) {
	    cerr << "Usage: aia3 <path to template image> [<path to test image>]" << endl;
	    cerr << "Press enter..." << endl;
	    cin.get();
	    return -1;
	}

	// construct processing object
	Aia3 aia3;
	
	if (argc == 2){
		// angle to rotate template image (in degree)
		float testAngle = 30;
		// factor to scale template image
		float testScale = 1.5;
		// run some test routines
		aia3.test(argv[1], testAngle, testScale);
	}else{
		// start processing
		aia3.run(argv[1], argv[2]);
	}
	
	return 0;

}
