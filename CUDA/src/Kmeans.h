/*
 * Kmeans.h
 *
 *  Created on: May 12, 2020
 *      Author: pietro
 */

#ifndef KMEANS_H_
#define KMEANS_H_

#include "Points.h"
#include "Centroids.h"
#include <cuda.h>

#define BLOCK_SIZE 128

void initPoints(PointsSet* points);

__host__ __device__ float distance(float& x1, float& x2, float& y1, float& y2);

void initCentroids(CentroidsSet* c, PointsSet* p);

__global__ void assignLabelKernel(PointsSet* points, CentroidsSet* centroids, int* differences);

void assingLabelsSequential(PointsSet* points, CentroidsSet* centroids, int* differences);

__global__ void updateCentroidsKernel(PointsSet* points, CentroidsSet* centroids, int* size);

void updateCentroidsSequential(PointsSet* points, CentroidsSet* centroids, int* size);

void parallelKmeans(PointsSet* points, CentroidsSet* centroids);

void sequentialKmeans(PointsSet* points, CentroidsSet* centroids);

#endif /* KMEANS_H_ */
