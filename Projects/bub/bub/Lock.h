#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
struct Lock
{
	int *mutex;
	__device__ Lock(void)
	{
#if __CUDA_ARCH__ >= 200
		mutex = new int;
		(*mutex) = 0;
#endif
	}
	__device__ ~Lock(void)
	{
#if __CUDA_ARCH__ >= 200
		delete mutex;
#endif
	}

	__device__ void lock(void)
	{
#if __CUDA_ARCH__ >= 200
		while (atomicCAS(mutex, 0, 1) != 0);
#endif
	}
	__device__ void unlock(void)
	{
#if __CUDA_ARCH__ >= 200
		atomicExch(mutex, 0);
#endif
	}
};