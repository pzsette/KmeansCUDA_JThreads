/*
 * Points.h
 *
 *  Created on: May 15, 2020
 *      Author: pietro
 */

#ifndef POINTS_H_
#define POINTS_H_

#include <iostream>

#define SIZE 500000

struct PointsSet {
	float x[SIZE];
	float y[SIZE];
	int labels[SIZE];
	int size;

	PointsSet() {
		size = 0;
	}

	void addPoint(float x1, float y1) {
		if (size < SIZE) {
			x[size] = x1;
			y[size] = y1;
			labels[size] = -1;
			size ++;
		}
	}

	void addPoint(float x1, float y1, int l) {
		if (size < SIZE) {
			x[size] = x1;
			y[size] = y1;
			labels[size] = l;
			size ++;
		}
	}

	void displayPoints() {
		printf("printo d_points");
		for(int i=0; i<size; i++) {
			printf("\nx[%d] = %f ; y[%d] = %f ; labels[%d] = %d\n", i, x[i], i, y[i], i, labels[i]);
		}
	}
};

#endif /* POINTS_H_ */
