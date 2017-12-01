//histogram processing

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t findFreqWithCuda(int *c, unsigned int fsize, const int *b, unsigned int size);




__global__ void addKernel(int *c, const int *b)
{
    int i = threadIdx.x;
	#if __CUDA_ARCH__ >= 200
	atomicAdd(&c[b[i]],1);
	#endif	
}

int main()
{
    const int arraySize = 11;
	const int fsize = 10005;//maximum value of the number in the array can be at max 10005
    const int a[arraySize] = { 1, 2, 3, 4, 5,1,1,1,2,3,6 };
	int maxx = INT_MIN;
	for (int i = 0; i < arraySize; i++)
	{
		//maxx = max(maxx, arr[i]);
		if (maxx < a[i])
			maxx=a[i];
	}
	int c[fsize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = findFreqWithCuda(c,fsize,a,arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "findFreqWithCuda failed!");
        return 1;
    }

	for (int i = 0; i <= maxx; i++)
	{
		if (c[i] != 0)
		{
			printf("%d occurs %d times\n", i, c[i]);
		}
	}

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
cudaError_t findFreqWithCuda(int *c, unsigned int fsize, const int *b, unsigned int size)
{
   // int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, fsize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
   

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	//dim3 threadsPerBlock(11,11,11);
    addKernel<<<1,size>>>(dev_c,dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, fsize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_b);
    
    return cudaStatus;
}
