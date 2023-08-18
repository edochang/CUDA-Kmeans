#pragma once

struct dataset_t {
    double*     points;
    double*     centers;
    double*     old_centers;
    int*        labels;
    int         points_size;
    int         point_rec_size;
    int         center_size;
    int         num_points;
    int         num_features;
    int         num_clusters;
    int         num_seed;
    double      threshold;
    int         max_num_iter;
    int         iter_to_converge;
    double      time_per_iter_in_ms;

    int getNumFeatures();
};

void fill_dataset(struct dataset_t* dataset, 
                  double*           points, 
                  int               points_size, 
                  int               point_rec_size,
                  int               num_points, 
                  int               num_features, 
                  int               num_clusters,
                  int               num_seed,
                  double            threshold,
                  int               max_num_iter);

int kmeans_rand();

void kmeans_srand(unsigned int seed);