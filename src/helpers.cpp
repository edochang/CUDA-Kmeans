#include "helpers.h"
#include <iostream>

static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

int dataset_t::getNumFeatures() {
    return num_features;
}

void fill_dataset(struct dataset_t* dataset, 
                  double*           points, 
                  int               points_size, 
                  int               point_rec_size,
                  int               num_points, 
                  int               num_features, 
                  int               num_clusters,
                  int               num_seed,
                  double            threshold,
                  int               max_num_iter) {
    //printf("helpers: fill dataset \n");  // debug statement

    dataset->points = points;
    //printf("helpers: Check last record of dataset array: %.15f \n", dataset->points[points_size-point_rec_size]);  // debug statement

    dataset->points_size = points_size;
    //printf("helpers: points size: %d \n", dataset->points_size);  // debug statement
    
    dataset->point_rec_size = point_rec_size;
    //printf("helpers: point record size: %d \n", dataset->point_rec_size);  // debug statement

    dataset->num_points = num_points;
    //printf("helpers: number of points: %d \n", dataset->num_points);  // debug statement

    dataset->num_features = num_features;
    //printf("helpers: number of features per point: %d \n", dataset->num_features);  // debug statement

    dataset->num_clusters = num_clusters;
    //printf("helpers: number of clusters: %d \n", dataset->num_clusters);  // debug statement

    dataset->num_seed = num_seed;
    //printf("helpers: seed: %d \n", dataset->num_seed);  // debug statement

    int center_size = num_clusters * num_features;
    dataset->center_size = center_size;
    //printf("helpers: center size: %d \n", dataset->center_size);  // debug statement

    // don't need if I'm creating the array on the heap within the code and passing the pointers / return them correctly.
    //dataset->centers = new double[center_size]; 
    //dataset->old_centers = new double[center_size];
    //dataset->old_centers = NULL;
    //printf("helpers: center and old_center arrays initialized \n");  // debug statement

    dataset->labels = new int[num_points];
    //printf("helpers: labels arrays initialized \n");  // debug statement

    dataset->threshold = threshold;
    //printf("helpers: threshold: %.15f \n", dataset->threshold);  // debug statement

    dataset->max_num_iter = max_num_iter;
    //printf("helpers: max_num_iter: %d \n", dataset->max_num_iter);  // debug statement
}

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}