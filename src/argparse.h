#pragma once

struct options_t {
    int num_clusters;
    int dims;
    char *input_filename;
    int max_num_iter;
    double threshold;
    bool c_flag;
    int seed;
    int records;
    int dataset_values;
    bool cpu;
    bool cuda;
    bool cuda_shmem;
    bool thrust;
};

void get_opts(int argc, char **argv, struct options_t *opts);