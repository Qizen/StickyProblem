/*
	Uses Utilities.h and .cpp for drawHistogram()
	and Images.cpp for invertImage()
	both from the book "A Practical Introduction to Computer Vision with OpenCV"
	by Kenneth Dawson-Howe
*/
#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime> //for timing the function
using namespace std;

Mat getHistogram(Mat& originalImage, Mat& mask, int numBins, bool showHist);
double getStdDev(Mat image, Mat mask);

int main(int argc, const char** argv)
{
	//time program execution
	time_t initTime = clock();

	//boolean array to store the results
	bool resultArray[6][5] = { false };
	bool trueValues[6][5] = {
		false, false, false, false, true, 
		false, true, false, false, false, 
		false, false, false, true, false, 
		false, true, true, false, false, 
		true, false, false, true, false, 
		false, false, false, false, true
	};


	// Loop through each image
	for(int i = 0; i<6; i++){
		char filename[50];
		//load up each image sequentially (glue1, glue2,...etc.)
		sprintf(filename, "glue%i.jpg", i+1);
		Mat image = imread(filename, -1);

		//imshow("test", image);
		waitKey(0); 
		// Convert to grayscale
		cvtColor(image, image, CV_RGB2GRAY);
		//imshow("converted", image);
		Mat highlightedGlue = image.clone();
		waitKey(0);

		Mat perPicMasks = Mat(image.rows, image.cols, CV_8UC1, Scalar(0.0));

		// Perform Gaussian blurring to get rid of (most of) the texture of the background
		Mat noiseReduced;
		GaussianBlur(image,noiseReduced,Size(5,5),1.5);
		//imshow("noise reduced", noiseReduced);

		// Threshold the noise reduced image to get rid of the background
		// Note -- Otsu thresholding doesn't work because it attempts to 
		// maximise the variance between white and black regions, which is undesirable.
		Mat thresholded;
		threshold(noiseReduced, thresholded, 50, 255, THRESH_TRIANGLE);
		//imshow("thresholded", thresholded);
		
		// Perform a very large (10 iterations) closing operation to get rid 
		// of areas within glue labels which are black.
		// This also closes in the gap between the shoulder of the bottle and 
		//the bottom of the lid, but this is small enough to be acceptable
		Mat finalGlueRegions;
		dilate(thresholded, thresholded, Mat(), Point(-1,-1), 10);
		erode(thresholded, finalGlueRegions, Mat(), Point(-1, -1), 10);
		//imshow("erode/dilate", finalGlueRegions);

		Mat mask = Mat(image.rows, image.cols, CV_8UC1, Scalar(0.0));

		for (int j = 0; j < 5; j++)
		{
			// zero the matrix
			mask = Mat::zeros(mask.rows, mask.cols, CV_8UC1); 
			
			/* 
			Create a mask which partitions the bottom half of the image into fifths
			this assumes that there are five glue bottles and that they're 
			evenly spaced throughout the bottom half of the image.

			Something could be done here with object recognition -- 
			determining the longest dimension of the glue bottles and then
			dividing them in two such that area is maximised should give
			a region roughly corresponding to the body of the glue bottle
			regardless of positioning and rotation.
			*/
			rectangle(mask, Point(j*mask.cols / 5, mask.rows / 2), Point((j+1)*mask.cols / 5, mask.rows), 
				Scalar(255), CV_FILLED, 8, 0);
			
			// invert and subtract the mask from the glue bottle regions determined earlier.
			Mat invertMask;
			invertImage(mask, invertMask);
			////imshow("MaskInverted", invertMask);
			Mat glueMask = Mat(image.rows, image.cols, CV_8UC1, Scalar(0.0));
			subtract(finalGlueRegions, invertMask, glueMask);
			//imshow("subtracted", glueMask);

			// add the mask to perPicMasks for showing later (for illustration -- not a part of the actual algo).
			add(perPicMasks, glueMask, perPicMasks);

			// create a histogram
			Mat hist = getHistogram(image, glueMask, 256, false);
			Mat histND[1]={hist};
			Mat histImage;
			Scalar mean, std_dev;

			// Calculate the std dev to differentiate between label and no label.
			meanStdDev(hist, mean, std_dev);
			// The following computes the mean and standard deviation of image greyscales rather than of the histogram
			// see report for more information about why we use the latter over the former.
			//meanStdDev(image, mean, std_dev, glueMask);
			DrawHistogram(histND, 1, histImage);

			// create a score value so that the result is not dependent on image size, multiplied by 1000
			// for no other reason than to have "nice" values.
			float score = std_dev[0]/countNonZero(glueMask)*1000;
			char outputChars[50];
			sprintf(outputChars, "score %i : %.4f mean : %.4f", j+1, score, mean[0]);
			putText(histImage, outputChars, Point(20, 20), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1, 8, false);
			char title[50];
			sprintf(title, "Glue Bottle #%i", j+1);
			//imshow(title, histImage); 
			if(score>7.0){
				resultArray[i][j] = true;
			}

		}
		////imshow("per pic masks", perPicMasks);
		Mat invertedPerPicMasks = Mat(image.rows, image.cols, CV_8UC1, Scalar(0.0));
		invertImage(perPicMasks, invertedPerPicMasks);
		////imshow("inverted per pic masks", invertedPerPicMasks);
		Mat selectedGlueBottleRegions = Mat(image.rows, image.cols, CV_8UC1, Scalar(0.0));
		subtract(image, invertedPerPicMasks, selectedGlueBottleRegions);
		//imshow("Selected regions", selectedGlueBottleRegions);
	}
	//waitKey(0);
	
	//Calculate time elapsed
	time_t elapsedTicks =  clock() - initTime;
	float elapsedTime = ((float)elapsedTicks)/CLOCKS_PER_SEC;
	// Calculate key metrics
	int TP = 0;
	int TN = 0;
	int FP = 0;
	int FN = 0;

	for(int i = 0; i<6; i++) {
		printf("Image %i:\n",i+1);
		for(int j = 0; j<5; j++){
			printf("Bottle %i : %i \t", j+1, resultArray[i][j]? 1 : 0);
			if(resultArray[i][j] == false && trueValues [i][j] == false)
				TN++;
			else if(resultArray[i][j] == true && trueValues [i][j] == true)
				TP++;
			else if(resultArray[i][j] == false && trueValues [i][j] == true)
				FN++;
			else if(resultArray[i][j] == true && trueValues [i][j] == false)
				FP++;
		}
		printf("\n");
	}
	printf("TP : %i \nTN : %i \nFP : %i \nFN : %i \n", TP, TN, FP, FN);
	float recall = TP/(TP+FN);
	float precision = TP/(TP+FP);
	float accuracy = (TP+TN)/30;
	float specificity = TN/(FP+TN);
	float F1 = 2*precision*recall/(precision+recall);
	printf("Recall : %.4f \nPrecision : %.4f \nAccuracy : %.4f \nSpecificity : %.4f \nF1 : %.4f\n", recall, precision, accuracy, specificity, F1);
	printf("Elapsed time: %.4f\n", elapsedTime);
	//waitKey() doesn't work without an imshow()
	getchar();
}

Mat getHistogram(Mat& originalImage, Mat& mask, int numBins, bool showHist){
	Mat hist;
	bool uniform = true;
	bool accumulate = false;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	calcHist(&originalImage, 1, 0, mask, hist, 1, &numBins, &histRange, uniform, accumulate);

	return hist;
}