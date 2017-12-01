//ye nhi chla sort
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Lock.h"

__global__ void addKernel(int *c)
{
	Lock ml;
	ml.lock();
    int i = threadIdx.x;
	int j = threadIdx.y;
	

	
	if ((i == j+1 && c[i] < c[j])|| (j == i+1 && c[i] > c[j]))
	{
			//exchange c[i] and c[j]
			//printf("c[i] : %d -- c[j] : %d\n", c[i], c[j]);


//#if __CUDA_ARCH__ >= 200
			int tempi = c[i];
			int tempj = c[j];
			c[i] = tempj;
			c[j] = tempi;
			//printf("i %d : j %d\n", i, j);
			//int z1 = atomicExch(&c[i], tempj);
			//int z2 = atomicExch(&c[j], tempi);
			//for (int i = 0; i < 5; i++)
			//{
			//	printf("%d ", c[i]);
			//}
			//printf("\n");
//#endif
			
	}
	ml.unlock();
}
cudaError_t addWithCuda(int *a, unsigned int size)
{
	int *dev_a = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(size, size);
	addKernel << <1, threadsPerBlock >> >(dev_a);

	cudaStatus = cudaGetLastError();
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_a);


	return cudaStatus;
}

int main()
{
    const int arraySize = 5;
    int a[arraySize] = {5,4,3,2,1};

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(a,arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("this is the sorted array = {%d,%d,%d,%d,%d}\n",
        a[0], a[1], a[2], a[3], a[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.

