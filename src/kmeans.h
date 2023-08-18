#pragma once

#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>

// Custom Libraries
#include "argparse.h"
#include "helpers.h"

// helper functions - prints
void printLabels_debug(int* labels, int num_points);
void printCenters(double* centers, int k, int dim);

// kmeans sequential
void* compute_kmeans_cpu(dataset_t* dataset);

// kmeans_kernel
void printClusterLabelCount(int* counts, int k);
void printConvergenceSum(double* convergence_k, int k);
bool isConverged(double* convergence_k, double threshold, int k);
void computeKMeansCuda(dataset_t* dataset);

// kmeans_kernel_shmem
void computeKMeansCudaShmem(dataset_t* dataset);

// kmeans_kernel_thrust
void computeKMeansThrust(dataset_t* dataset);