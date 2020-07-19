/*
 * Centroids.h
 *
 *  Created on: May 16, 2020
 *      Author: pietro
 */

#ifndef CENTROIDS_H_
#define CENTROIDS_H_

#include <stdio.h>

#define K 5

struct CentroidsSet {
	float x[K];
	float y[K];
	int labels[K];
	int size=0;

	CentroidsSet() {
		size = 0;
	}

	void addCentroid(float x1, float y1) {
		if (size < SIZE) {
			x[size] = x1;
			y[size] = y1;
			size ++;
		}
	}

	void addCentroid(float x1, float y1, int l) {
		if (size < SIZE) {
			x[size] = x1;
			y[size] = y1;
			labels[size] = l;
			size ++;
		}
	}

	void displayCentroids() {
		printf("size centroids:%d", size);
		for(int i=0; i<size; i++) {
			printf("\nx[%d] = %f ; y[%d] = %f ; labels[%d] = %d\n", i, x[i], i, y[i], i, labels[i]);
		}
	}
};

#endif /* CENTROIDS_H_ */
