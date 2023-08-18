#include "kmeans.h"
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK(call) {                                                           \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess) {                                                 \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                           \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
        exit(1);                                                                \
    }                                                                           \
}                                                                               \

void printClusterLabelCount(int* counts, int k) {
    printf("print cluster label count: ");
    for (int i = 0; i < k; i++){
        printf(" %d", counts[i]);
    }
    printf("\n");
}

void printConvergenceSum(double* convergence_k, int k) {
    printf("print convergence sum: ");
    for (int i = 0; i < k; i++){
        printf(" %lf", convergence_k[i]);
    }
    printf("\n");
}

bool isConverged(double* convergence_k, double threshold, int k) {
    bool converge_result = true;
    for (int i = 0; i < k; i++) {
        if (convergence_k[i] > threshold) {
            converge_result = false;
            return converge_result;
        }
    }
    return converge_result;
}

__global__ void findNearestCentroidsCudaKernel(double* points, double* centers, int* labels, int dim, int k, int num_points, int threads) {
    //printf("hello world from block %d and thread %d with dim(%d), k(%d) \n", blockIdx.x, threadIdx.x, dim, k);  // debug statement

    int point_label_index = threadIdx.x + blockIdx.x * threads;
    
    if (point_label_index < num_points) {
        int offset = dim + 1;
        int point_index = (threadIdx.x + blockIdx.x * threads) * offset;
        int point_index_dim_start = point_index + 1;
        
        int cluster_label;
        double min;
        for (int clusterId = 0; clusterId < k; clusterId++) {
            double dis = 0.0;
            double sum = 0.0;
            for (int d = 0; d < dim; d++) {
                double point_diff = points[point_index_dim_start + d] - centers[(clusterId * dim) + d];
                sum = sum + (point_diff * point_diff);
            }
            dis = sqrt(sum);

            if (clusterId == 0) {
            min = dis;
            cluster_label = clusterId;
            } else {
                if (dis < min) {
                    min = dis;
                    cluster_label = clusterId;
                }
            }
        }
        
        labels[point_label_index] = cluster_label;

        // debug statements
        /*
        if (point_label_index == 60) {
            printf("Print Points: ");
            for (int d = 0; d < dim; d++) {
                printf(" %lf", points[point_index_dim_start + d]);
            }
            printf("\n");
            
            for (int clusterId = 0; clusterId < k; clusterId++) {
                printf("%d ", clusterId);
                for (int d = 0; d < dim; d++) {
                    printf("%lf ", centers[(clusterId * dim) + d]);
                }
                printf("\n");
            }
            
            printf("findNearestCentroidsCudaK: label(%d): cluster_label: %d \n", point_label_index, cluster_label);
            printf("findNearestCentroidsCudaK: label(%d): %d \n", point_label_index, labels[point_label_index]);
        }
        */
    }
}

__global__ void addLabeledCentroidsCudaKernel(double* points, double* centers, int* labels, int* cluster_label_count, int dim, int num_points, int threads) {
    int pointId = threadIdx.x + blockIdx.x * threads;
    
    if (pointId < num_points) {
        int clusterId = labels[pointId];
        int offset = dim + 1;  // includes point index value so offset by 2 to get first dim element index if point
        int points_start_index = pointId * offset + 1;
        int center_start_index = clusterId * dim;

        for (int d = 0; d < dim; d++) {
            atomicAdd(&centers[center_start_index + d], points[points_start_index + d]);
        }
        atomicAdd(&cluster_label_count[clusterId], 1);
    }
}

__global__ void averageLabeledCentroidsCudaKernel(double* centers, int* cluster_label_count, int dim, int k, int threads) {
    int clusterId = threadIdx.x + blockIdx.x * threads;

    if (clusterId < k) {
        int cluster_start_index = clusterId * dim;
        for (int d = 0; d < dim; d++) {
            centers[cluster_start_index + d] = centers[cluster_start_index + d] / cluster_label_count[clusterId];
        }
    }
}

__global__ void convergenceCudaKernel(double* centers, double* old_centers, double* convergence_k, int dim, int k, int threads) {
    int clusterId = threadIdx.x + blockIdx.x * threads;
    
    if (clusterId < k) {
        double cluster_converge_sum = 0;
        int ci_start = (clusterId * dim);
        for (int d = 0; d < dim; d++) {
            cluster_converge_sum = cluster_converge_sum + ((centers[ci_start + d] - old_centers[ci_start + d]) / old_centers[ci_start + d] * 100);
        }
        convergence_k[clusterId] = abs(cluster_converge_sum);

        //printf("Hello from thread(%d)! \n", clusterId);  // debug statement;
    }
}

void computeKMeansCuda(dataset_t* dataset) {
    //printf("kmeans: Compute K Means Cuda \n");  // debug statement
    
    // create timers
    float memsettime;
    cudaEvent_t start_iter, stop_iter;
    cudaEventCreate(&start_iter);
    cudaEventCreate(&stop_iter);
    
    // can be commented out when not taking fine-grain measurements.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // use to calculate E2E execution time
    //cudaEventRecord(start, 0);

    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    // book-keeping
    int iteration = 0;
    bool done = false;
    double iter_time = 0;
    bool converged = false;

    // host variables
    // shared variables
    int k = dataset->num_clusters;
    int dim = dataset->num_features;
    int num_points = dataset->num_points;
    int points_size = dataset->points_size;
    double* points = dataset->points;
    int* labels = dataset->labels;
    int* cluster_label_count = new int[k] { 0 };
    double* centers = dataset->centers;  // gets the random centroids from main
    double* old_centers = new double[k * dim] { 0 };
    double* convergence_k = new double[k] { 0 };

    //printCenters(centers, k, dim);  // debug statement
    //printf("computeKMeansCuda: center_size: %d \n", dataset->center_size);  // debug statement
    //printf("computeKMeansCuda: points_size: %d \n", points_size);  // debug statement
 
    // set threads per block
    int threads = 256;
    int blocks_points = (num_points + threads - 1 ) / threads;  // divide by points instead.
    int blocks_cluster = (k + threads - 1 ) / threads;  // divide by clusters instead.
    //printf("computeKMeansCuda: blocks_points: %d \n", blocks_points);  // debug statement
    //printf("computeKMeansCuda: blocks_cluster: %d \n", blocks_cluster);  // debug statement

    //printf("computeKMeansCuda: blocks: %d \n", blocks);  // debug statement
    //int blocks = (points_size + threads - 1 ) / threads;  // divide event by elements which includes points and dimensions.

    // device variables
    size_t size_bytes_points = points_size * sizeof(double);
    size_t size_bytes_centers = k * dim * sizeof(double);
    size_t size_bytes_labels = num_points * sizeof(int);
    size_t size_bytes_cluster_label_count = k * sizeof(int);
    size_t size_bytes_convergence_k = k * sizeof(double);

    //printf("computeKMeansCuda: size_bytes_points: %ld \n", size_bytes_points);  // debug statement

    double *d_points, *d_centers, *d_old_centers, *d_convergence_k;
    int *d_labels, *d_cluster_label_count;

    //printCenters(centers, k, dim);  // debug statement

    //cudaEventRecord(start, 0);
    cudaMalloc((void**) &d_points, size_bytes_points);
    cudaMalloc((void**) &d_centers, size_bytes_centers);
    cudaMalloc((void**) &d_labels, size_bytes_labels);
    cudaMalloc((void**) &d_cluster_label_count, size_bytes_cluster_label_count);
    cudaMalloc((void**) &d_convergence_k, size_bytes_convergence_k);  // needed for convergence
    cudaMalloc((void**) &d_old_centers, size_bytes_centers);  // needed for convergence
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&memsettime, start, stop);
    //printf("*** CUDA execution time (cudaMalloc for 5): %f *** \n", memsettime);

    //cudaEventRecord(start, 0);
    cudaMemcpy(d_points, points, size_bytes_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, centers, size_bytes_centers, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, size_bytes_labels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster_label_count, cluster_label_count, size_bytes_cluster_label_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_convergence_k, convergence_k, size_bytes_convergence_k, cudaMemcpyHostToDevice);
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&memsettime, start, stop);
    //printf("*** CUDA execution time (cudaMemcpy for 5): %f *** \n", memsettime);

    while(!done) {
    //while(iteration != 1) {
        cudaEventRecord(start_iter, 0);
        iteration++;

        // ****** findNearestCentroidsCudaKernel ****** //
        //cudaEventRecord(start, 0);
        findNearestCentroidsCudaKernel<<<blocks_points,threads>>>(d_points, d_centers, d_labels, dim, k, num_points, threads);
        //cudaEventRecord(stop, 0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&memsettime, start, stop);
        //printf("*** CUDA execution time (findNearestCentroidsCudaKernel): %f *** \n", memsettime);

        //CHECK(cudaDeviceSynchronize());

        /*
        // debug statement
        cudaMemcpy(labels, d_labels, size_bytes_labels, cudaMemcpyDeviceToHost);
        printLabels_debug(labels, num_points);  // debug statement
        */

        // ****** averageLabeledCentroidsCudaKernel ****** //
        cudaMemcpy(d_old_centers, d_centers, size_bytes_centers, cudaMemcpyDeviceToDevice);

        /*
        // debug statement
        printf("print old centers(%d): \n", iteration);
        cudaMemcpy(old_centers, d_old_centers, size_bytes_centers, cudaMemcpyDeviceToHost);
        printCenters(old_centers, k, dim);  // debug statement
        */

        cudaMemset(d_centers, 0, size_bytes_centers);
        cudaMemset(d_cluster_label_count, 0, size_bytes_cluster_label_count);
        
        /*
        // debug statement
        printf("print 0 centers \n");
        cudaMemcpy(centers, d_centers, size_bytes_centers, cudaMemcpyDeviceToHost);
        printCenters(centers, k, dim);  // debug statement
        */

        //cudaEventRecord(start, 0);
        addLabeledCentroidsCudaKernel<<<blocks_points,threads>>>(d_points, d_centers, d_labels, d_cluster_label_count, dim, num_points, threads);
        //cudaEventRecord(stop, 0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&memsettime, start, stop);
        //printf("*** CUDA execution time (addLabeledCentroidsCudaKernel): %f *** \n", memsettime);

        //CHECK(cudaDeviceSynchronize());

        /*
        // debug statement
        printf("print added centers \n");
        cudaMemcpy(centers, d_centers, size_bytes_centers, cudaMemcpyDeviceToHost);
        printCenters(centers, k, dim);  // debug statement
        
        // debug statement
        printf("print cluster_label_count \n");
        cudaMemcpy(cluster_label_count, d_cluster_label_count, size_bytes_cluster_label_count, cudaMemcpyDeviceToHost);
        printClusterLabelCount(cluster_label_count, k);
        */

        //cudaEventRecord(start, 0);
        averageLabeledCentroidsCudaKernel<<<blocks_cluster,threads>>>(d_centers, d_cluster_label_count, dim, k, threads);
        //cudaEventRecord(stop, 0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&memsettime, start, stop);
        //printf("*** CUDA execution time (averageLabeledCentroidsCudaKernel): %f *** \n", memsettime);

        //CHECK(cudaDeviceSynchronize());
        
        //cudaMemcpy(centers, d_centers, size_bytes_centers, cudaMemcpyDeviceToHost);
        //printCenters(centers, k, dim);  // debug statement

        // ****** convergenceCudaKernel ****** //
        
        cudaMemset(d_convergence_k, 0, size_bytes_convergence_k);

        /*
        // debug statement
        cudaMemcpy(convergence_k, d_convergence_k, size_bytes_convergence_k, cudaMemcpyDeviceToHost);
        printConvergenceSum(convergence_k, k);  // debug statement
        */

        //cudaEventRecord(start, 0);
        convergenceCudaKernel<<<blocks_cluster,threads>>>(d_centers, d_old_centers, d_convergence_k, dim, k, threads);
        //cudaEventRecord(stop, 0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&memsettime, start, stop);
        //printf("*** CUDA execution time (convergenceCudaKernel): %f *** \n", memsettime);
        
        //cudaEventRecord(start, 0);
        cudaMemcpy(convergence_k, d_convergence_k, size_bytes_convergence_k, cudaMemcpyDeviceToHost);
        //cudaEventRecord(stop, 0);
        //cudaEventSynchronize(stop);
        //cudaEventElapsedTime(&memsettime, start, stop);
        //printf("*** CUDA execution time (cudaMemcpy for 1): %f *** \n", memsettime);
        
        //printConvergenceSum(convergence_k, k);  // debug statement

        CHECK(cudaDeviceSynchronize());

        converged = isConverged(convergence_k, dataset->threshold, k);

        done = iteration > dataset->max_num_iter || converged;

        cudaEventRecord(stop_iter, 0);
        cudaEventSynchronize(stop_iter);
        cudaEventElapsedTime(&memsettime, start_iter, stop_iter);
        iter_time = iter_time + memsettime;
    }
    //printf("computeKMeansCuda: done? %s \n", done ? "true" : "false");
    dataset->time_per_iter_in_ms = iter_time / (double)iteration;
    dataset->iter_to_converge = iteration;
    //printf("computeKMeansCuda: time_per_iter_in_ms: %lf, iteration: %d \n", dataset->time_per_iter_in_ms, dataset->iter_to_converge);
    
    //cudaEventRecord(start, 0);
    cudaMemcpy(labels, d_labels, size_bytes_labels, cudaMemcpyDeviceToHost);
    cudaMemcpy(old_centers, d_old_centers, size_bytes_centers, cudaMemcpyDeviceToHost);
    cudaMemcpy(centers, d_centers, size_bytes_centers, cudaMemcpyDeviceToHost);
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&memsettime, start, stop);
    //printf("*** CUDA execution time (cudaMemcpy for 3): %f *** \n", memsettime);

    // use to calculate E2E execution time
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&memsettime, start, stop);
    //printf("*** CUDA execution time (E2E): %f *** \n", memsettime);

    dataset->labels = labels;
    dataset->centers = centers;
    dataset->old_centers = old_centers;

    cudaFree(d_points);
    cudaFree(d_centers);
    cudaFree(d_old_centers);
    cudaFree(d_convergence_k);
    cudaFree(d_labels);
    cudaFree(d_cluster_label_count);

    cudaEventDestroy(start_iter);
    cudaEventDestroy(stop_iter);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
