#include <argparse.h>

// https://www.gnu.org/software/libc/manual/html_node/Getopt.html
#include <getopt.h>
#include <iostream>

// Flag set by --long_options
static int cpu_flag;
static int cuda_flag;
static int cuda_shmem_flag;
static int thrust_flag;

void get_opts(int               argc,
              char              **argv,
              struct options_t  *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <an integer specifying the number of clusters>" << std::endl;
        std::cout << "\t--dims or -d <an integer specifying the dimension of the points>" << std::endl;
        std::cout << "\t--inputfilename or -i <a string specifying the input filename>" << std::endl;
        std::cout << "\t--max_num_iter or -m <an integer specifying the maximum number of iterations>" << std::endl;
        std::cout << "\t--threshold or -t <a double specifying the threshold for convergence test>" << std::endl;
        std::cout << "\t--seed or -s <an integer specifying the seed for rand(). This is used by the autograder to simplify the correctness checking process>" << std::endl;
        std::cout << "\t--use_cpu to do a sequential execution" << std::endl;
        std::cout << "\t--use_cuda to run cuda basic execution" << std::endl;
        std::cout << "\t--use_cuda_s to run cuda shmem execution" << std::endl;
        std::cout << "\t--use_thrust to run thrust execution" << std::endl;
        std::cout << "\t[Optional] --centroid_output_flag or -c <a flag to control the output of the program. If -c is specified, this program will output the centroids of all clusters. If -c is not specified, this program will output the labels of all points>" << std::endl;
        exit(0);
    }

    // Set flag values.
    opts->c_flag = false;
    opts->cpu = false;
    opts->cuda = false;
    opts->cuda_shmem = false;
    opts->thrust = false;

    struct option l_opts[] = {
        {"num_cluster", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"inputfilename", required_argument, NULL, 'i'},
        {"max_num_iter", required_argument, NULL, 'm'},
        {"threshold", required_argument, NULL, 't'},
        {"seed", required_argument, NULL, 's'},
        {"centroid_output_flag", no_argument, NULL, 'c'},
        {"use_cpu", no_argument, &cpu_flag, 1},
        {"use_cuda", no_argument, &cuda_flag, 1},
        {"use_cuda_s", no_argument, &cuda_shmem_flag, 1},
        {"use_thrust", no_argument, &thrust_flag, 1}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:s:c", l_opts, &ind)) != -1)
    {
        //printf("argparse: c value: %d \n", c);  // debug statement
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->num_clusters = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'i':
            opts->input_filename = (char *)optarg;
            break;
        case 'm':
            opts->max_num_iter = atoi((char *)optarg);;
            break;
        case 't':
            opts->threshold = atof((char *)optarg);
            break;
        case 'c':
            opts->c_flag = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            //printf("argparse: option seed: %d", opts->seed);  // debug statement
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(0);
        }
    }

    if (cpu_flag) {
        opts->cpu = true;
        //printf("argparse: cpu flag triggered \n");  // debug statement
    }

    if (cuda_flag) {
        opts->cuda = true;
        //printf("argparse: cuda flag triggered \n");  // debug statement
    }

    if (cuda_shmem_flag) {
        opts->cuda_shmem = true;
        //printf("argparse: cuda_shmem flag triggered \n");  // debug statement
    }

    if (thrust_flag) {
        opts->thrust = true;
        //printf("argparse: thrust flag triggered \n");  // debug statement
    }
}
