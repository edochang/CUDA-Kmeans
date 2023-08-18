# KMeans with CUDA
From the KMeans with CUDA Lab, I observed my CUDA Basic implementation to be the fastest for 2k (k16, d16) points and 16k (k16, d24) points input.  I also observed my CUDA Shared Memory implementation to be the fastest for 65k (k16, d32) points input.  See the figure below.

Based on these observations, CUDA Basic and CUDA Shared Memory met my expectations on which should be faster depending on record size.  

![GPU_Speedup_by_Implementation](/markdown_assets/GPU_Speedup_by_Implementation.png)

![Average_Iteration_Time_by_Implementation](/markdown_assets/Average_Iteration_Time_by_Implementation.png)

For Shared Memory, I chose the centers to be kept in Shared Memory, because of how much this data is accessed multiple times to find nearest centroid, calculating averages, and calculating convergences in my implementation.  I did not choose points and their dimensions due to concerns with the Shared Memory limit per block.  

Below is an example of my hypothesis notes for Shared Memory for Centers and Dimensions vs. Points and Dimensions

> The most read / accessed data here is the centers as each point is compared to each center to find the nearest centroid.  Total shared memory per block for a RTX 3060 Laptop GPU is 49,152 bytes (from deviceQuery).  80% of this is 39,322 bytes.  39,322 bytes / 8 bytes (per double) is ~4,900 double elements that can be allocated per block.  32*16 center elements are 512 elements.  Assumption is that we will not be handling so many clusters and dimensions (features) for calculating KMean, so it’s a good candidate for implementing Shared Memory.  Using points and dimensions wouldn’t be possible because the bytes needed exceeds the Share Memory size (2,048 points * 16 dimensions * 8-byte doubles = 262,144 bytes).

By applying Shared Memory on the centers and their elements, I noticed improve read speed and as a result an improvement to the average iteration times for CUDA Shared Memory.  This is a benefit of exposing Memory Hierarchy by the GPU Architecture, so we can utilize faster reads within a block between threads. 

Shared Memory however did not beat 2k (k16, d16) points and 16k (k16, d24) points input.  I speculate it’s because the time and extra operational steps to read and copy the data from global memory to shared memory by a set of threads in the block is not economical for processing smaller points / inputs.  

CUDA Basic and CUDA Shared Memory met my expectations on who’s slower based on the number of data / records that’s being processed.  CUDA Shared Memory is slower in this Lab for smaller datasets (e.g., < ~65k points) and CUDA Basic is slower for larger dataset (e.g., > ~65k points).  

The Lab2 instructions hinted Thrust will be slower but was not able to observe this as I had a difficult time understanding the programming model, managing iterators, and setting up the right shaped vectors.  Based on what I read and was able to understand from the Thrust libraries, a lot of the fine-grain tuning is not exposed to the programmer like CUDA Basic and Shared Memory and is managed by the thrust library, for example Memory Hierarchy settings, Block and Thread allocation settings, etc.  I speculate Thrust implementation will be faster than sequential / CPU implementation but would not beat CUDA Basic or Shared Memory.

Though I was not able to complete the full KMeans for Thrust, I was able to complete similar execution with finding nearest centroids.  The Thrust execution took a little longer than the kernel execution version in CUDA Basic.  To a certain degree my speculation is directionally correct thus far.

![Comparing_Thrust_and_Cuda_Find_Nearest_Centroids_Execution_Time_for_65k_Points](/markdown_assets/Comparing_Thrust_and_Cuda_Find_Nearest_Centroids_Execution_Time_for_65k_Points.png)

## Best-Case Performance
Best-Case Performance will depend on how much of the GPU’s architecture can be utilized for parallel execution.  For a 1D configuration, my RTX 3060 Laptop GPU can support a grid size of 2,147,483,647 blocks and a thread block of 1024 threads.  

The table below shows what happens if we adjust the threads per block to 128, 256, or 512 for a CUDA Basic implementation.  The times are average iteration times.

![CUDA_Basic_Block_Adjustment_Table_1](/markdown_assets/CUDA_Basic_Block_Adjustment_Table_1.png)

Based on the general structure of my parallel implementation, I speculate I can improve it in the following ways to further get the best-case performance by utilizing as much of the GPU Architecture’s Blocks and Threads.
- If algorithm permits for labeling points with nearest centroid, decompose from points to dimension-features to fill the blocks and thread to utilize more of the processors and memory on the GPU Architecture.
- If algorithm permits for adding centroid sum from labeled points, average new centroid, and convergence calculations, further decompose the work from centroid to dimensions within the centroid.  This will enable the use of more blocks and threads from 16 centers to 512 (k - 16 * d – 32) center dimension-features.

For a 65k point input, assuming points is the best decomposition, I speculate I can get a potential of 0.02 ms (0.01 ms from each kernel that is currently decomposed by centers) improvements in decomposing the centroid to its dimension-features.  Currently with 16 centers, I am only using 1 block and 16 threads with idling threads in the block.  This is a perfect example of the taco stations examples explained in the lecture where the benefits of using GPU Architecture is to fill and use as much of the high memory bandwidth and SM processors that the architecture offers to really capitalize on the benefits the architecture offers.

*** CUDA execution time (findNearestCentroidsCudaKernel): 2.438112 ***

*** CUDA execution time (addLabeledCentroidsCudaKernel): 0.617472 ***

*** CUDA execution time (averageLabeledCentroidsCudaKernel): 0.025600 ***

*** CUDA execution time (convergenceCudaKernel): 0.024576 ***

Based on my sequential implementation of KMeans I speculate the best-case performance speedup will be around ~30 to ~40 percent speed-up for data points greater than 16k.  This is close to what I observed in my CUDA Basic implementation.

## Data Transfer Overheads
Depending on the number of data that’s being processed / executed by the thread block, the fraction of the end-to-end runtime is anywhere from ~2% (65k points) to ~33% (with 2k points).  This shows the importance of memory management as another point for optimization considerations. 

![Data_Transfer_Overhead_Table_2](/markdown_assets/Data_Transfer_Overhead_Table_2.png)

The table below shows the ratio of transferring data from host-to-device and device-to-host one time by the CUDA Basic implementation (outside the while-loop iteration).   Here we can see it takes ~1% (65k) to ~6% (2k) of the E2E run time.  If we compare the two Data Transfer / E2E between one-time memory transfers plus transfers done in the while-loop and one-time memory transfer only, we can observe the impact (increased fraction / ratio) to the execution time if we execute too many data transfers between host and device.  Striking the right balance and keeping to a minimum the need to transfer data between host and device can help optimize the parallel execution on the device side.  

![Data_Transfer_Overhead_Table_3](/markdown_assets/Data_Transfer_Overhead_Table_3.png)

Note:  Times are captured in milliseconds (ms).  These are for CUDA Basic with a block size of 256 threads per block.

The more we can leave our data on the Device side to complete our operations the better the execution time.  One reason to trade this off, is if we need to do global synchronization of the data and apply operations that’s required to run on host device before being able to proceed with more parallel execution.

# Appendix
## TO DO
- Complete Thrust implementation

## Additional Observations
### 3 Observation Samples for Each Implementation and Input
![Additional_Observations_Table_1](/markdown_assets/Additional_Observations_Table_1.png)

### Observation Averages
![Additional_Observations_Table_2](/markdown_assets/Additional_Observations_Table_2.png)


### Memory Allocation Timing / Observation
Used CUDA events to time on GPU side to transfer / copy data to device.

![Additional_Observations_Table_3](/markdown_assets/Additional_Observations_Table_3.png)

### cudaDeviceSynchronize() After Every Kernel Run
This synchronization is only needed if we need to block the host to wait for CUDA devices to sync.  Did not need to run this after every kernel since there was no Host calculation / operation.  Only area that needed this was when I tested for Convergence.  Saw speed up when this was removed from the implementation.
The graphs below can be compared to the main graphs above in the main report.

![GPU_Speedup_by_Implementation_2](/markdown_assets/GPU_Speedup_by_Implementation_2.png)
![Average_Iteration_Time_by_Implementation_2](/markdown_assets/Average_Iteration_Time_by_Implementation_2.png)

## Computer Information
Used local development environment using Windows Subsystem Linux 2 (WSL2).

### CPU Hardware
AMD Ryzen 9 5900HS with Radeon Graphics, 3301 Mhz, 8 Core(s), 16 Logical Processor(s)
Installed Physical Memory (RAM): 24.0 GB

### GPU Hardware
![GPU Hardware](/markdown_assets/Computer_Info.png)

### OS Version
Windows 11 Pro, Version 22H2, OS Build 22621.608
Windows Subsystem for Linux Distributions:  Ubuntu-20.04
uname -r >>> 5.15.57.1-microsoft-standard-WSL2

### Environment Notes
I used atomicAdd operations in my implementation using double data types.  Double data types are only supported on hardware with architectures above sm_60.  In my Makefile, I used -arch=native to compile my code.  Native will derive the current GPU architecture used by the device.

Example:
nvcc **-arch=native** ./src/argparse.cpp ./src/helpers.cpp ./src/io.cpp ./src/kmeans.cpp ./src/kmeans_kernel_shmem.cu ./src/kmeans_kernel.cu ./src/kmeans_thrust.cu -std=c++17 -I ./src/ -o bin/kmeans

## Running The Program
./bin/kmeans -k 16 -d 16 -i /home/username/dev/repo/Lab2/tests/input/random-n2048-d16-c16.txt -m 150 -t 0.000001 -s 8675309 -c --use_cpu
    
**./bin/kmeans argument instructions**

    Usage:
        --num_cluster or -k <an integer specifying the number of clusters>
        --dims or -d <an integer specifying the dimension of the points>
        --inputfilename or -i <a string specifying the input filename>
        --max_num_iter or -m <an integer specifying the maximum number of iterations>
        --threshold or -t <a double specifying the threshold for convergence test>
        --seed or -s <an integer specifying the seed for rand(). This is used by the autograder to simplify the correctness checking process>
        --use_cpu to do a sequential execution
        --use_cuda to run cuda basic execution
        --use_cuda_s to run cuda shmem execution
        --use_thrust to run thrust execution
        [Optional] --centroid_output_flag or -c <a flag to control the output of the program. If -c is specified, this program will output the centroids of all clusters. If -c is not specified, this program will output the labels of all points>