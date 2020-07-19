
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <stdlib.h>
#include <sstream>
#include <string>
#include <cuda.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include "Kmeans.h"
#include "utils.cuh"

void initPoints(PointsSet* p) {
		std::ifstream file;
		file.open("data.txt");
		std::string line;
		if (file.is_open()) {
			while (getline(file, line)) {
				std::replace(line.begin(), line.end(), ',', ' ');
				std::vector<float> array;
				std::stringstream ss(line);
				float temp;
				while (ss >> temp){
					array.push_back(temp);
				}
				p->addPoint(array[0],array[1]);
			}
			file.close();
		}else{
			printf("Error");
		}
}

__host__ __device__ float distance(float& x1, float& x2, float& y1, float& y2) {
	float distance = (float)sqrt(pow(y2 - y1,2) + pow(x2 - x1,2));
	return distance;
}

void initCentroids(CentroidsSet* c, PointsSet* p) {
	std::vector<int> indexes;
	srand(time(NULL));
	int count=0;
	while (count<K) {
		int rnd = rand()%p->size;
		if (std::find(indexes.begin(), indexes.end(), rnd)== indexes.end()) {
			indexes.push_back(rnd);
			c->addCentroid(p->x[rnd], p->y[rnd], -1);
			count++;
		}
	}
}

__global__ void assingLabelsKernel(PointsSet* points, CentroidsSet* centroids, int* updates) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < points->size) {
		float minDistance = distance(points->x[index], centroids->x[0], points->y[index], centroids->y[0]);
		int minIndex = 0;
		int j;
		float tmpDistance;
		for (j=1; j<centroids->size; j++) {
			tmpDistance = distance(points->x[index], centroids->x[j], points->y[index], centroids->y[j]);
			if (tmpDistance < minDistance) {
				minDistance = tmpDistance;
				minIndex = j;
			}
		}
		if (points->labels[index] != minIndex) {
			atomicAdd(updates, 1);
		}
		points->labels[index] = minIndex;
	}
}

__global__ void updateCentroidsKernel(PointsSet* points, CentroidsSet* centroids, int* size) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < points->size) {
		atomicAdd(&(centroids->x[points->labels[index]]), points->x[index]);
		atomicAdd(&(centroids->y[points->labels[index]]), points->y[index]);
		atomicAdd(&(size[points->labels[index]]), 1);
	}
}

void assignLabelSequential(PointsSet* points, CentroidsSet* centroids, int* updates) {
	for (int index=0; index<points->size; index++) {
		float minDistance = distance(points->x[index], centroids->x[0], points->y[index], centroids->y[0]);
		int minIndex = 0;
		int j;
		float tmpDistance;
		for (j=1; j<centroids->size; j++) {
			tmpDistance = distance(points->x[index], centroids->x[j], points->y[index], centroids->y[j]);
			if (tmpDistance < minDistance) {
				minDistance = tmpDistance;
				minIndex = j;
			}
		}
		if (points->labels[index] != minIndex) {
			(*updates)++;
		}
		points->labels[index] = minIndex;
	}
}

void updateCentroidsSequential(PointsSet* points, CentroidsSet* centroids, int* size) {
	for (int index=0; index<points->size; index++) {
			centroids->x[points->labels[index]] += points->x[index];
			centroids->y[points->labels[index]] += points->y[index];
			size[points->labels[index]] += 1;
	}
}

void parallelKmeans(PointsSet* points, CentroidsSet* centroids) {
	printf("Parallel execution with CUDA\n");
	int* nClusters = (int*)malloc(K*sizeof(int));
	int* updates = (int*)malloc(K*sizeof(int));

	PointsSet* d_points;
	CentroidsSet* d_centroids;
	int* d_nClusters;
	int* d_updates;

	CUDA_CHECK_RETURN(cudaMalloc((void **) &d_points, sizeof(PointsSet)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &d_centroids, sizeof(CentroidsSet)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &d_nClusters, K*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void **) &d_updates, sizeof(int)));

	initPoints(points);
	initCentroids(centroids, points);

	CUDA_CHECK_RETURN(cudaMemcpy(d_points, points, sizeof(PointsSet), cudaMemcpyHostToDevice));

	auto start = std::chrono::high_resolution_clock::now();

	do {
		(*updates)=0;
		CUDA_CHECK_RETURN(cudaMemcpy(d_updates, updates, sizeof(int), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_centroids, centroids, sizeof(CentroidsSet), cudaMemcpyHostToDevice));

		assingLabelsKernel<<<ceil((double)(points->size)/BLOCK_SIZE), BLOCK_SIZE>>>(d_points, d_centroids, d_updates);
		cudaDeviceSynchronize();

		for (int i=0; i<K; i++) {
			centroids->x[i] = 0;
			centroids->y[i] = 0;
			centroids->labels[i] = -1;
			nClusters[i] = 0;
		}

		CUDA_CHECK_RETURN(cudaMemcpy(d_centroids, centroids, sizeof(CentroidsSet), cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(d_nClusters, nClusters, K*sizeof(int), cudaMemcpyHostToDevice));

		updateCentroidsKernel<<<ceil((double)(points->size)/BLOCK_SIZE), BLOCK_SIZE>>>(d_points, d_centroids, d_nClusters);
		cudaDeviceSynchronize();

		CUDA_CHECK_RETURN(cudaMemcpy(centroids, d_centroids, sizeof(CentroidsSet), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(nClusters, d_nClusters, K*sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(updates, d_updates, sizeof(int), cudaMemcpyDeviceToHost));

		for (int i=0; i<K; i++) {
			centroids->x[i] = centroids->x[i]/nClusters[i];
			centroids->y[i] = centroids->y[i]/nClusters[i];
			centroids->labels[i] = i;
		}

		printf("Number of updates:%d\n", (*updates));


	} while (*updates != 0);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end-start;
	printf("Time parallel execution: %f\n", time.count());

	cudaMemcpy(points, d_points, sizeof(PointsSet), cudaMemcpyDeviceToHost);

	CUDA_CHECK_RETURN(cudaFree(d_points));
	CUDA_CHECK_RETURN(cudaFree(d_centroids));
	CUDA_CHECK_RETURN(cudaFree(d_nClusters));
	CUDA_CHECK_RETURN(cudaFree(d_updates));

	free(updates);
	free(nClusters);


}

void sequentialKmeans(PointsSet* points, CentroidsSet* centroids){
	printf("Sequential execution\n");
	int* nClusters = (int*)malloc(K*sizeof(int));
	int* updates = (int*)malloc(sizeof(int));

	initPoints(points);
	initCentroids(centroids, points);

	auto start = std::chrono::high_resolution_clock::now();

	do {
		(*updates) = 0;

		assignLabelSequential(points, centroids, updates);

		for (int i=0; i<K; i++) {
			centroids->x[i] = 0;
			centroids->y[i] = 0;
			centroids->labels[i] = -1;
			nClusters[i] = 0;
		}

		updateCentroidsSequential(points, centroids, nClusters);

		for (int i=0; i<K; i++) {
			centroids->x[i] = centroids->x[i]/nClusters[i];
			centroids->y[i] = centroids->y[i]/nClusters[i];
			centroids->labels[i] = i;
		}

		printf("Number of updates:%d\n", *updates);


	} while (*updates != 0);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end-start;
	printf("Time sequential execution: %f", time.count());

	free(nClusters);
	free(updates);

}




