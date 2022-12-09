#include <stdio.h>
#include <stdint.h>

#include "support.h"
#include "kernel.cu"

//#include<iostream>

int main(int argc, char* argv[])
{
    Timer timer;
    //timeval time;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    //startTime(&timer);

    unsigned int* in_h;
    unsigned int* bins_h;
    unsigned int* in_d;
    unsigned int* bins_d;
    unsigned int num_elements, num_bins;
    cudaError_t cuda_ret;

    if (argc == 1) {
        num_elements = 10000000;
        num_bins = 1024;
    }
    else if (argc == 2) {
        num_elements = atoi(argv[1]);
        num_bins = 4096;
    }
    else if (argc == 3) {
        num_elements = atoi(argv[1]);
        num_bins = atoi(argv[2]);
    }
    else {
        printf("\n    Invalid input parameters!"
            "\n    Usage: ./histogram            # Input: 1,000,000, Bins: 4,096"
            "\n    Usage: ./histogram <m>        # Input: m, Bins: 4,096"
            "\n    Usage: ./histogram <m> <n>    # Input: m, Bins: n"
            "\n");
        exit(0);
    }
  /*  in_h = (unsigned int*)malloc(num_elements * sizeof(unsigned int));
    bins_h = (unsigned int*)malloc(num_bins * sizeof(unsigned int));*/

    cuda_ret = cudaHostAlloc((void**)&in_h, num_elements * sizeof(unsigned int), cudaHostAllocDefault);
    cuda_ret = cudaHostAlloc((void**)&bins_h, num_bins * sizeof(unsigned int), cudaHostAllocDefault);

    initVector(&in_h, num_elements, num_bins);


    //stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of bins = %u\n", num_elements,
        num_bins);

    //// Allocate device variables ----------------------------------------------

    //printf("Allocating device variables..."); fflush(stdout);
    ////startTime(&timer);
    //gettimeofday(&time);
    //timeval s = time;

    //cuda_ret = cudaMallocManaged((void**)&in_d, num_elements * sizeof(unsigned int));
    //if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
    //cuda_ret = cudaMallocManaged((void**)&bins_d, num_bins * sizeof(unsigned int));
    //if (cuda_ret != cudaSuccess) printf("Unable to allocate device memory");

    //cudaDeviceSynchronize();
    //gettimeofday(&time);
    //timeval e = time;
    //printf("\n Time %f \n",((float)((e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec) / 1.0e6)));
    ////stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    //// Copy host variables to device ------------------------------------------

    //printf("Copying data from host to device..."); fflush(stdout);
    ////startTime(&timer); 
    //gettimeofday(&time); s = time;

    //cuda_ret = cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int),
    //    cudaMemcpyHostToDevice);
    //if (cuda_ret != cudaSuccess) printf("Unable to copy memory to the device");

    //cuda_ret = cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int));
    //if (cuda_ret != cudaSuccess) printf("Unable to set device memory");

    //cudaDeviceSynchronize();
    ////stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    //gettimeofday(&time); e = time;
    //printf("\n Time %f \n", ((float)((e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec) / 1.0e6)));


    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    //gettimeofday(&time); timeval s = time;

    histogram(in_h, bins_h, num_elements, num_bins);
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) printf("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("This includes - Device Allocation time,  Host to Device transfer time, Kernel time and Device to Host transfer time");
   /* gettimeofday(&time); timeval e = time;
    printf("\n Time %f \n", ((float)((e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec) / 1.0e6)));*/


    //// Copy device variables from host ----------------------------------------
    //gettimeofday(&time); s = time;

    //printf("Copying data from device to host..."); fflush(stdout);
    ////startTime(&timer);

    //cuda_ret = cudaMemcpy(bins_h, bins_d, num_bins * sizeof(unsigned int),
    //    cudaMemcpyDeviceToHost);
    //if (cuda_ret != cudaSuccess) printf("Unable to copy memory to host");

    //cudaDeviceSynchronize();
    ////stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    //gettimeofday(&time); e = time;
    //printf("\n Time %f \n", ((float)((e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec) / 1.0e6)));


    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, bins_h, num_elements, num_bins);

    // Free memory ------------------------------------------------------------

    //cudaFree(in_d); 
    // cudaFree(bins_d);
    cudaFreeHost(in_h); cudaFreeHost(bins_h);

    return 0;
}
