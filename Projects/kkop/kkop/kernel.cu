//matrix multiplication


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define N 3
__global__ void MatMul(int A[][N], int B[][N], int C[][N]) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int temp;
	for (int j1 = 0; j1 < N; j1++)
	{
		temp = A[i][j] * B[j][j1];
		#if __CUDA_ARCH__ >= 200
		atomicAdd(&C[i][j1], temp);
		#endif
	}
}


int main() {

	int A[N][N] = { { 1,2,3 },{ 4,5,6 },{ 7,8,9 } };
	int B[N][N] = { { 1,2,3 },{ 4,5,6 },{ 7,8,9 } };
	int C[N][N] = { { 0,0,0 },{ 0,0,0 },{ 0,0,0 } };

	int(*pA)[N], (*pB)[N], (*pC)[N], (*pf)[N];

	cudaMalloc((void**)&pA, (N*N) * sizeof(int));
	cudaMalloc((void**)&pB, (N*N) * sizeof(int));
	cudaMalloc((void**)&pC, (N*N) * sizeof(int));

	cudaMemcpy(pA, A, (N*N) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, B, (N*N) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, C, (N*N) * sizeof(int), cudaMemcpyHostToDevice);
	int numBlocks = 1;
	dim3 threadsPerBlock(N,N);
	MatMul << <numBlocks, threadsPerBlock >> >(pA, pB, pC);

	cudaMemcpy(C, pC, (N*N) * sizeof(int), cudaMemcpyDeviceToHost);

	int i, j; printf("\nC = \n");
	for (i = 0; i<N; i++) {
		for (j = 0; j<N; j++) {
			printf("%d ", C[i][j]);
		}
		printf("\n");
	}

	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);

	printf("\n");

	return 0;
}