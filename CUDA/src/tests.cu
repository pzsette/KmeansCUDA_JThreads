/*
 ============================================================================
 Name        : kmeansCUDA.cu
 Author      : pietro
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include "Kmeans.h"
#include "Points.h"
#include "Centroids.h"
#include <stdio.h>




int main(int argc, char** argv)
{
	//Initialize for random number generation
	srand(time(NULL));

	PointsSet* p = (PointsSet*)malloc(sizeof(PointsSet));
	CentroidsSet* c = (CentroidsSet*)malloc(sizeof(CentroidsSet));

	//Run sequential execution of Kmeans algorithm
	//sequentialKmeans(p, c);

	//Run parallel execution of Kmeans alorithm with CUDA
	parallelKmeans(p, c);


	//Save points label in "output.txt" file
	/*FILE *output;
	int i;
	output = fopen("output.txt", "w");
	for (i=0; i<p->size; i++) {
		fprintf(output, "%f,%f,%d\n", p->x[i], p->y[i], p->labels[i]);
	}*/

	return 0;
}

