#include<stdio.h>


#define BLOCK_SIZE 512
#define GRID_SIZE 150
__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	/*************************************************************************/
	// INSERT KERNEL CODE HERE
	__shared__ unsigned int bin_private[4096];

	// initialize bin counters to 0
	if (threadIdx.x < num_bins)
		bin_private[threadIdx.x] = 0;
	int n = threadIdx.x + BLOCK_SIZE;
	while (n < num_bins)
	{
		bin_private[n] = 0;
		n += BLOCK_SIZE;
	}

	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// stride is total number of threads
	int stride = blockDim.x * gridDim.x;
	while (i < num_elements) {
		/*if (input[i] == 0)
			printf("\n count \n");*/
		atomicAdd(&(bin_private[input[i]]), 1);
		i += stride;
	}

	__syncthreads();

	if (threadIdx.x < num_bins)
		atomicAdd(&(bins[threadIdx.x]), bin_private[threadIdx.x]);

	n = threadIdx.x + BLOCK_SIZE;
	while (n < num_bins)
	{
		atomicAdd(&(bins[n]), bin_private[n]);
		n += BLOCK_SIZE;
	}

	/*************************************************************************/
}

__global__ void vecadd_kernel(unsigned int* intermediate_bin, unsigned int* bins, unsigned int num_bins)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < num_bins)
	{
		bins[i] += intermediate_bin[i];
	}
}

//void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {
//
//	/*************************************************************************/
//  //INSERT CODE HERE
//
//	dim3 DimGrid(GRID_SIZE, 1, 1);
//	dim3 DimBlock(BLOCK_SIZE, 1, 1);
//
//	histo_kernel << <DimGrid, DimBlock >> > (input, bins, num_elements, num_bins);
//
//	/*************************************************************************/
//
//}

void histogram(unsigned int* a_h, unsigned int* bins_h, unsigned int num_elements, unsigned int num_bins) {

	/*************************************************************************/
  //INSERT CODE HERE
	int segSize = 100000;
	if (num_elements < 2*segSize)
		segSize = num_elements / 2;
	
	cudaStream_t stream0;
	cudaStream_t stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	cudaError_t cudaStatus;

	unsigned int* a_d0, * a_d1; // ARRAY
	unsigned int* b_d0, * b_d1; // bins for streams
	unsigned int* bins_d; // actual bins

	cudaStatus = cudaMallocManaged((void**)&a_d0, segSize * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc failed!");
	cudaStatus = cudaMallocManaged((void**)&a_d1, segSize * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc failed!");

	cudaStatus = cudaMalloc((void**)&b_d0, num_bins * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc failed!");
	cudaStatus = cudaMalloc((void**)&b_d1, num_bins * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc failed!");

	cudaStatus = cudaMallocManaged((void**)&bins_d, num_bins * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMalloc failed!");
	cudaStatus = cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) printf("Unable to set device memory");

	cudaStatus = cudaMemset(b_d0, 0, num_bins * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) printf("Unable to set device memory");
	cudaStatus = cudaMemset(b_d1, 0, num_bins * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) printf("Unable to set device memory");
	

	for (int i = 0; i < num_elements; i += segSize * 2)
	{
		cudaMemcpyAsync(a_d0, a_h + i, segSize * sizeof(unsigned int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(a_d1, a_h + i + segSize, segSize * sizeof(unsigned int), cudaMemcpyHostToDevice, stream1);
		histo_kernel << <GRID_SIZE, BLOCK_SIZE, 0, stream0 >> > (a_d0, b_d0, segSize, num_bins);
		histo_kernel << <GRID_SIZE, BLOCK_SIZE, 0, stream1 >> > (a_d1, b_d1, segSize, num_bins);
	}

	vecadd_kernel << <((num_bins - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0, stream0 >> > (b_d0, b_d1, num_bins);


	cudaMemcpy(bins_h, b_d1, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	
	/*************************************************************************/

}
