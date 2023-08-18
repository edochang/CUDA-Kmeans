#include "kmeans.h"
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

struct find_nearest_centroids_f : public thrust::unary_function<double, int> {
    int k;
    int dim;
    double* points;
    double* centers;

    explicit find_nearest_centroids_f(int _k, int _dim, double* _centers, double* _points) : 
        k(_k), dim(_dim), centers(_centers), points(_points) {};

    __host__ __device__
    double operator()(const int& x) {
        int offset = dim + 1;
        int point_index = x * offset;
        int point_index_dim_start = point_index + 1;
        
        int cluster_label;
        double min;
        double sum = 0.0;
        for (int clusterId = 0; clusterId < k; clusterId++) {
            double dis = 0.0;
            sum = 0.0;
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

        /*
        printf("x(%d) \n", x);

        if (x == 0) {
            printf("Print Points: ");
            for (int d = 0; d < dim; d++) {
                printf(" %lf", points[point_index_dim_start + d]);
            }
            printf("\n");
        }
        */

        return cluster_label;
    }
};

/* Comment out as it's not working.  TO DO Area.
struct add_labeled_centroids_f : public thrust::unary_function<void, int> {
    int k;
    int dim;
    double* points;
    double* centers;
    int* labels;

    explicit add_labeled_centroids_f(int _k, int _dim, double* _centers, double* _points, int* _labels) : 
        k(_k), dim(_dim), centers(_centers), points(_points), labels(_labels) {};
    
    __host__ __device__
    void operator()(const int& x) {
        int pointId = x;
        int clusterId = labels[pointId];
        int offset = dim + 1;  // includes point index value so offset by 2 to get first dim element index if point
        int points_start_index = pointId * offset + 1;
        int center_start_index = clusterId * dim;

        for (int d = 0; d < dim; d++) {
            atomicAdd(&centers[center_start_index + d], points[points_start_index + d]);
        }
    }
};
*/

struct remove_points_index : public thrust::unary_function<double, int> {
    int k;
    int dim;

    remove_points_index(int _k, int _dim) : k(_k), dim(_dim) {}

    __host__ __device__
    bool operator()(const int& x) {
        return x % (dim + 1) == 0;
    }
};

void computeKMeansThrust(dataset_t* dataset) {
    //printf("kmeans: Compute K Means Cuda \n");  // debug statement
    // book-keeping
    int iteration = 0;
    bool done = false;
    double iter_time = 0;
    //bool converged = false;

    // create timers
    float memsettime;
    cudaEvent_t start_iter, stop_iter;
    cudaEventCreate(&start_iter);
    cudaEventCreate(&stop_iter);
    
    // host variables
    // shared variables
    int k = dataset->num_clusters;
    int dim = dataset->num_features;
    int num_points = dataset->num_points;
    int points_size_with_index = dataset->points_size;
    //int points_size = num_points * dim;
    int center_size = dataset->center_size;
    double* points = dataset->points;
    int* labels = dataset->labels;
    int* cluster_label_count = new int[k] { 0 };
    double* centers = dataset->centers;  // gets the random centroids from main
    double* old_centers = new double[k * dim] { 0 };
    double* convergence_k = new double[k] { 0 };
    
    printf("computeKMeansThrust: points_size_with_index: %d \n", points_size_with_index);
    
    // convert to vectors
    thrust::device_vector<double> vector_centers(centers, centers + center_size);
    thrust::device_vector<double> vector_old_centers(old_centers, old_centers + center_size);
    thrust::device_vector<double> vector_points_dim(points, points + points_size_with_index);
    thrust::device_vector<int> vector_labels(labels, labels + num_points);

    thrust::device_vector<int> vector_cluster_label_count(k, 0);

    //printf("vector_labels size: %zu \n", vector_labels.size());

    // debug statement
    //thrust::copy(vector_points_dim.begin(), vector_points_dim.end(), std::ostream_iterator<double>(std::cout, " "));
    //printf("\n");

    // helper vectors
    thrust::device_vector<int> vector_points_index(num_points);
    thrust::device_vector<double> vector_points_cluster_sum_then_sqrt(num_points * k);
    thrust::device_vector<int> vector_labels_sort_key(num_points);
    thrust::device_vector<int> vector_cluster_label_ones(num_points, 1);
    thrust::device_vector<int> vector_cluster_label_count_keys(k);

    thrust::sequence(vector_points_index.begin(), vector_points_index.end());

    // debug statement
    //thrust::copy(vector_points_index.begin(), vector_points_index.end(), std::ostream_iterator<double>(std::cout, " "));
    //printf("\n");

    /*
    thrust::device_vector<double> vector_points_new(points_size);

    remove_points_index removePointsIndex(k, dim);
    //thrust::copy_if(thrust::device, vector_points_with_index.begin(), vector_points_with_index.begin() + points_size_with_index, vector_points.begin(), isPointsElementPtsIndex);
    vector_points_index.erase(thrust::remove_if(thrust::device, vector_points_index.begin(), vector_points_index.end(), removePointsIndex), vector_points_index.end());
    
    thrust::copy_n(thrust::device,
        thrust::make_permutation_iterator(vector_points.begin(), vector_points_index.begin()),
        vector_points_new.size(),
        vector_points_new.begin()
    );
    */

    // debug statement
    //thrust::copy(vector_points_index.begin(), vector_points_index.end(), std::ostream_iterator<double>(std::cout, " "));
    //printf("\n");

    

    // debug statement
    //thrust::copy(vector_points_new.begin(), vector_points_new.end(), std::ostream_iterator<double>(std::cout, " "));
    //printf("\n");

    //while(!done) {
    while(iteration != 1) {
        cudaEventRecord(start_iter, 0);
        iteration++;

        //auto start = std::chrono::high_resolution_clock::now();

        // ****** findNearestCentroidsCudaKernel ****** //
        thrust::transform(vector_points_index.begin(), vector_points_index.end(), vector_labels.begin(), find_nearest_centroids_f(k, dim, thrust::raw_pointer_cast(vector_centers.data()), thrust::raw_pointer_cast(vector_points_dim.data())));

        //auto end = std::chrono::high_resolution_clock::now();
        //auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);     
        //printf("kmeans(cpu): time_per_iter_in_ms: %lf \n", (double)diff.count());

        // debug statement
        //thrust::copy(vector_labels.begin(), vector_labels.end(), std::ostream_iterator<double>(std::cout, " "));
        //printf("\n");

        //cudaDeviceSynchronize();

        // ****** averageLabeledCentroidsCudaKernel ****** //

        thrust::copy(thrust::device, vector_centers.begin(), vector_centers.end(), vector_old_centers.begin());
        
        thrust::fill(vector_cluster_label_count.begin(), vector_cluster_label_count.end(), 0);
        thrust::copy(thrust::device, vector_labels.begin(), vector_labels.end(), vector_labels_sort_key.begin());
        thrust::stable_sort(vector_labels_sort_key.begin(), vector_labels_sort_key.end());
        
        thrust::reduce_by_key(vector_labels_sort_key.begin(), vector_labels_sort_key.end(), vector_cluster_label_ones.begin(), vector_cluster_label_count_keys.begin(), vector_cluster_label_count.begin());
        
        // debug statement
        //printf("vector_labels_sort_key: ");
        //thrust::copy(vector_labels_sort_key.begin(), vector_labels_sort_key.end(), std::ostream_iterator<double>(std::cout, " "));
        //printf("\n");

        /* Comment out as it's not working.  TO DO Area.
        thrust::fill(vector_centers.begin(), vector_centers.end(), 0);
        thrust::for_each(vector_points_index.begin(), vector_points_index.end(), 
            add_labeled_centroids_f(
                k, 
                dim, 
                thrust::raw_pointer_cast(vector_centers.data()), 
                thrust::raw_pointer_cast(vector_points_dim.data()),
                thrust::raw_pointer_cast(vector_labels.data())
            ));
        */

        // debug statement
        printf("vector_centers: ");
        thrust::copy(vector_centers.begin(), vector_centers.end(), std::ostream_iterator<double>(std::cout, " "));
        printf("\n");
        printf("vector_cluster_label_count: ");
        thrust::copy(vector_cluster_label_count.begin(), vector_cluster_label_count.end(), std::ostream_iterator<double>(std::cout, " "));
        printf("\n");

        //averageLabeledCentroidsCudaKernel_shmem<<<blocks_cluster, threads, size_bytes_centers>>>(d_centers, d_cluster_label_count, dim, k, threads);
        
        //printConvergenceSum(convergence_k, k);  // debug statement

        //converged = isConverged(convergence_k, dataset->threshold, k);

        //done = iteration >= dataset->max_num_iter || converged;

        cudaEventRecord(stop_iter, 0);
        cudaEventSynchronize(stop_iter);
        cudaEventElapsedTime(&memsettime, start_iter, stop_iter);
        iter_time = iter_time + memsettime;
    }
    printf("computeKMeansCuda: done? %s \n", done ? "true" : "false");
    dataset->time_per_iter_in_ms = iter_time / (double)iteration;
    dataset->iter_to_converge = iteration;
    printf("computeKMeansCuda: time_per_iter_in_ms: %lf, iteration: %d \n", dataset->time_per_iter_in_ms, dataset->iter_to_converge);
    
    //dataset->labels = labels;
    //dataset->centers = centers;
    //dataset->old_centers = old_centers;

    cudaEventDestroy(start_iter);
    cudaEventDestroy(stop_iter);
}

/*
    Thrust Notes
    thrust::for_each - Iterates through a vector and applying a functor for each element.  (https://nvidia.github.io/thrust/api/groups/group__modifying.html)
    
    thrust::stable_sort_by_key - sort two key-value vectors by its key.  (https://nvidia.github.io/thrust/api/groups/group__sorting.html#function-stable-sort-by-key)
    
    thrust::reduce_by_key - reduce the values of the consecutive equal keys.  You can set your own binary predicate condition and binary operator.  (https://nvidia.github.io/thrust/api/groups/group__reductions.html#function-reduce-by-key)
    
    thrust::transform - This version of transform applies a unary function to each element of an input sequence and stores the result in the corresponding position in an output sequence.  (https://nvidia.github.io/thrust/api/groups/group__transformations.html#function-transform)
    
    thrust::upper_bound - Does a binary search on the vector of ordered range (first, last), it return the index of the last position where value could be inserted without violating ordering.   (https://nvidia.github.io/thrust/api/groups/group__vectorized__binary__search.html#function-upper-bound) - note this one has a special signature for comp operator.

    https://nvidia.github.io/thrust/api/groups/group__copying.html

*/