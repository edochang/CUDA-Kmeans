#include "kmeans.h"
#include <iomanip>
#include <algorithm>

// Custom Libraries
#include "argparse.h"
#include "io.h"
#include "helpers.h"

using namespace std;

void printCenters(double* centers, int k, int dim) {
    /* 
    if -c is specified, your program should output the centroids of final clusters in the following form

        for (int clusterId = 0; clusterId < _k; clusterId ++) {
            printf("%d ", clusterId);
            for (int d = 0; d < _d; d++)
                printf("%lf ", centers[clusterId + d * _k]);
            printf("\n");
        }

    Note that the first column is the cluster id.
    */
    for (int clusterId = 0; clusterId < k; clusterId++) {
        printf("%d ", clusterId);
        for (int d = 0; d < dim; d++) {
            printf("%lf ", centers[(clusterId * dim) + d]);
            //printf("%.5f ", centers[(clusterId * dim) + d]);
        }
        printf("\n");
    }
}

void printLabels_debug(int* labels, int num_points) {
    printf("clusters:");
    for(int p = 0; p < num_points; p++) {
        printf("p[%d]%d ", p, labels[p]);
    }
    printf("\n");
}

void printLabels(int* labels, int num_points) {
    /*  If -c is not specified to your program, it needs to write points assignment, i.e. the final cluster id for each point, to STDOUT in the following form:
            
            printf("clusters:")
            for (int p=0; p < nPoints; p++)
                printf(" %d", clusterId_of_point[p]);

        The autograder will redirect your output to a temp file, which will be further processed to check correctness.
    */
    printf("clusters:");
    for(int p = 0; p < num_points; p++) {
        printf(" %d", labels[p]);
    }
    printf("\n");
}

void printTime(int iter_to_converge, double time_per_iter_in_ms) {
    /*  
    For each input, each implementation is asked to classify the points into k clusters. You should measure the elapsed time and total number of iterations for Kmeans to converge. The averaged elapsed time per iteration and the number of iterations to converge should be written to STDOUT in the following form
    
    Note that the time should be measured in milliseconds. Part of your grade will be based on how fast your implementation is.
    */
    printf("%d,%lf\n", iter_to_converge, time_per_iter_in_ms);
}

double* randomCentroids(dataset_t* dataset, int dim, int k) {
    double* points = dataset->points;
    int center_size = dim * k;
    double* centers = new double[center_size]; 

    int _numpoints = dataset->num_points;
    int point_rec_size = dataset->point_rec_size;

    //printf("kmeans: _numpoints: %d \n", _numpoints);  // debug statement

    // calculate initial centers
    for (int i=0; i<k; i++) {
        int index = kmeans_rand() % _numpoints;
        // you should use the proper implementation of the following
        // code according to your data structure
        index = index * point_rec_size;
        int point_1st_dim = index + 1;
        int ci = i * dim;
        centers[ci] = i;
        
        //printf("kmeans: ci value: %d \n", ci);  // debug statement
        //printf("kmeans: random index: %d ", index);  // debug statement
        //printf(", random index value: %.15f \n", points[index]);  // debug statement
        for(int j = point_1st_dim; j < point_1st_dim + dim; ++j) {
            //printf("kmeans: j value: %d ", j);  // debug statement
            //printf(", j value: %.15f \n", points[j]);  // debug statement
            centers[ci] = points[j];
            ++ci;
        }
    }

    return centers;
}

int* findNearestCentroids(dataset_t* dataset, double* centers) {
    double* points = dataset->points;
    int* labels = dataset->labels;
    int num_points = dataset->num_points;
    int point_rec_size = dataset->point_rec_size;
    int k = dataset->num_clusters;
    int dim = dataset->num_features;
    
    //double point[point_rec_size];

    for(int i = 0; i < num_points; i++) {
        /*
        // store the dimensions of a point into an array to represent it
        for (int p = 0; p < point_rec_size; p++) {
            point[p] = points[(i * point_rec_size) + p];  
            //printf("kmeans: p: %d ", p);  // debug statement
            //printf(", p value: %.15f \n", point[p]);  // debug statement
        }
        */

        int point_index = i * point_rec_size;
        int point_index_dim_start = point_index + 1;

        // calculate euclidian distance against the point's dimensions compared to the dimensions of the other clusters.
        int cluster_label;
        double min;
        for (int clusterId = 0; clusterId < k; clusterId++) {
            // Calculate euclidean distance against the current cluster
            double dis = 0.0;
            double sum = 0.0;
            for (int d = 0; d < dim; d++) {
                //sum = sum + pow((points[point_index_dim_start + d]-centers[(clusterId * dim) + d]), 2.0);
                double point_diff = points[point_index_dim_start + d] - centers[(clusterId * dim) + d];
                sum = sum + (point_diff * point_diff);
            }
            dis = sqrt(sum);
            //printf("kmeans: euclidiean distance: %.15f \n", dis);  // debug statement

            // Set the minimum distance with the first cluster
            if (clusterId == 0) {
                min = dis;
                cluster_label = clusterId;
            } else {
                // compare newly calculated distance against the minimum.  If less than the minimum, set as the new minimum.
                if (dis < min) {
                    min = dis;
                    cluster_label = clusterId;
                }
            }
        }

        // if (i==0) { printf("kmeans: minimum distance for p(%d) to cluster(%d): %.15f \n", i, cluster_label, min); } // debug statement

        // set the label for the point index.
        labels[i] = cluster_label;
    }

    return labels;
}

double* averageLabeledCentroids(dataset_t* dataset, int* labels) {
    double* points = dataset->points;
    double* centers = new double[dataset->center_size] { 0 };
    int num_points = dataset->num_points;
    int point_rec_size = dataset->point_rec_size;
    int k = dataset->num_clusters;
    int dim = dataset->num_features;
    
    int center_label_size[k] = { 0 };

    //printf("kmeans: Check center_label_size value for index(%d): %d \n", 6, center_label_size[6]);  // debug statement
    //printf("kmeans: averageLabeledCentroid - num_points: %d \n", num_points);  // debug statement
    //printf("kmeans: averageLabeledCentroid - point_rec_size: %d \n", point_rec_size);  // debug statement
    //printf("kmeans: averageLabeledCentroid - k: %d \n", k);  // debug statement

    // calculate sum for each cluster for labeled points.
    for (int label = 0; label < num_points; label++) {
        int clusterId = labels[label];
        int center_start_index = clusterId * dim;
        int point_start = label * point_rec_size + 1;
        for (int d = 0; d < dim; d++) {
            centers[center_start_index + d] = centers[center_start_index + d] + points[point_start + d];
        }
        center_label_size[clusterId] = center_label_size[clusterId] + 1;
    }

    /* debug statement
    printf("kmeans: label_dim_sum for cluster(%d) with size(%d): ", clusterId, center_label_size[clusterId]);
    for (int d = 0; d < dim; d++) {
        printf("%.15f ", label_dim_sum[d]);
    }
    printf("\n");
    */

    for (int clusterId = 0; clusterId < k; clusterId++) {
        int center_start_index = clusterId * dim;
        for (int d = 0; d < dim; d++) {
            int ci = center_start_index + d;
            centers[ci] = centers[ci] / center_label_size[clusterId];
        }
    }

    //fill(label_dim_sum, label_dim_sum + dim, 0);
    
    /* debug statement - Check if label_dim_sum is filled with 0.
    printf("kmeans: label_dim_sum for cluster(%d) with size(%d): ", clusterId, center_label_size[clusterId]);
    for (int d = 0; d < dim; d++) {
        printf("%.15f ", label_dim_sum[d]);
    }
    printf("\n");
    */

    return centers;
}

bool converged(dataset_t* dataset, double* centers, double* old_centers) {
    int k = dataset->num_clusters;
    int dim = dataset->num_features;
    double threshold = dataset->threshold;
    bool converge_result = true;

    for (int clusterId = 0; clusterId < k; clusterId++) {
        double cluster_converge_sum = 0;
        int ci_start = (clusterId * dim);
        for (int d = 0; d < dim; d++) {
            cluster_converge_sum = cluster_converge_sum + ((centers[ci_start + d] - old_centers[ci_start + d]) / old_centers[ci_start + d] * 100);
        }

        cluster_converge_sum = abs(cluster_converge_sum);

        if (cluster_converge_sum > threshold) {
            converge_result = false;
            break;
        } else {
            converge_result = true;
        }
    }

    return converge_result;
}

void* compute_kmeans_cpu(dataset_t* dataset) {
    //printf("kmeans: Compute K Means CPU \n");  // debug statement
    int* labels;

    double* centers = dataset->centers;

    // book-keeping
    int iteration = 0;
    double* old_centers = NULL;
    double iter_time = 0;

    // core algorithm
    bool done = false;
    
    // Start timer
    while(!done) {
        auto start = std::chrono::high_resolution_clock::now();
        if (old_centers != NULL) {
            delete[] old_centers;
        }
        old_centers = centers;
        iteration++;

        // labels is a mapping from each point in the dataset 
        // to the nearest (euclidean distance) centroid
        labels = findNearestCentroids(dataset, centers);

        centers = averageLabeledCentroids(dataset, labels);

        done = iteration > dataset->max_num_iter || converged(dataset, centers, old_centers);

        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        iter_time = iter_time + (double)diff.count();

        //if (iteration == 1) {  printLabels_debug(labels, dataset->num_points); }  // debug statement
        //if (iteration == 1) {  printCenters(centers, dataset->num_clusters, dataset->num_features); }  // debug statement
    }
    //End timer
    dataset->time_per_iter_in_ms = iter_time / (double)iteration;
    dataset->iter_to_converge = iteration;
    //printf("kmeans(cpu): time_per_iter_in_ms: %lf, iteration: %d *** \n", dataset->time_per_iter_in_ms, dataset->iter_to_converge);

    dataset->old_centers = old_centers;
    dataset->centers = centers;

    return 0;
}

int main(int argc, char **argv)
{
    // Parse args
    struct options_t opts;
    get_opts(argc, argv, &opts);

    //cout << "main: Running the program: " << argv[0] << endl;  // debug statement
    //cout << "main: Read input file: " << opts.input_filename << endl;  // debug statement

    // Read input data
    double* points;
    read_file(&opts, &points);

    int point_rec_size = opts.dims + 1;
    int point_size = opts.records * point_rec_size;

    // Fill Dataset Struct
    struct dataset_t dataset;
    fill_dataset(&dataset, points, point_size, point_rec_size, opts.records, opts.dims, opts.num_clusters, opts.seed, opts.threshold, opts.max_num_iter);

    // set seed for random generator
    int cmd_seed = dataset.num_seed;
    kmeans_srand(cmd_seed);  // cmd_seed is a cmdline arg

    // initialize centroid randomly
    int k = dataset.num_clusters;
    int dim = dataset.getNumFeatures();
    dataset.centers = randomCentroids(&dataset, dim, k);

    //printf("kmeans.main: opts.cpu: %s \n", opts.cpu ? "true" : "false");  // debug statement

    if (opts.cpu) {
        compute_kmeans_cpu(&dataset);
        
        printTime(dataset.iter_to_converge, dataset.time_per_iter_in_ms);

        //printf("main: opts.c_flag: %s \n", opts.c_flag ? "true" : "false");  // debug statement
        if (opts.c_flag) {
            // if -c is specified, your program should output the centroids of final clusters.
            printCenters(dataset.centers, k, dim);
        } else {
            /*  
            If -c is not specified to your program, it needs to write points assignment.
            The autograder will redirect your output to a temp file, which will be further processed to check correctness.
            */
            printLabels(dataset.labels, dataset.num_points);
        }

        // reduce by key and for each will be helpful for thrust.
        
        // Free arrays
        //printf("main: freeing global arrays in the heap \n");  // debug statement
    }

    if (opts.cuda) {
        computeKMeansCuda(&dataset);

        printTime(dataset.iter_to_converge, dataset.time_per_iter_in_ms);
        if (opts.c_flag) {
            printCenters(dataset.centers, k, dim);
        } else {
            printLabels(dataset.labels, dataset.num_points);
        }
    }

    if (opts.cuda_shmem) {
        computeKMeansCudaShmem(&dataset);

        printTime(dataset.iter_to_converge, dataset.time_per_iter_in_ms);
        if (opts.c_flag) {
            printCenters(dataset.centers, k, dim);
        } else {
            printLabels(dataset.labels, dataset.num_points);
        }
    }

    if (opts.thrust) {
        computeKMeansThrust(&dataset);
    }
    
    // reduce by key and for each will be helpful for thrust.

    // Free arrays
    //printf("main: freeing global arrays in the heap \n");  // debug statement
    delete[] dataset.points;
    delete[] dataset.centers;
    delete[] dataset.old_centers;
    delete[] dataset.labels;

    return 0;
}