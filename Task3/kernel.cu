#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dev_array.h"

__global__
void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

	int ROW = blockIdx.y*blockDim.y + threadIdx.y;
	int COL = blockIdx.x*blockDim.x + threadIdx.x;

	if (ROW < N && COL < N) {
		float tmpSum = 0.0f;
		// each thread computes the block sub-matrix
		for (int i = 0; i < N; i++) {
			tmpSum += A[ROW * N + i] * B[i * N + COL];
		}
		C[ROW * N + COL] = tmpSum;
	}
}


void matrixMultiplication(float *A, float *B, float *C, int N) {

	// declare the number of blocks per grid and the number of threads per block
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N*N > 512) {
		threadsPerBlock.x = 512;
		threadsPerBlock.y = 512;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerBlock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerBlock.y));
	}

	matrixMultiplicationKernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, N);
}

int main() {
	std::chrono::steady_clock::time_point start, stop;
	std::chrono::microseconds duration;

	int N = 1024;
	int SIZE = N * N;

	// host memory allocation
	std::vector<float> h_A(SIZE);
	std::vector<float> h_B(SIZE);
	std::vector<float> h_C(SIZE);

	// initialize matrices
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			h_A[i*N + j] = 128.5f;
			h_B[i*N + j] = 256.25f;
		}
	}

	// device memory allocation
	dev_array<float> d_A(SIZE);
	dev_array<float> d_B(SIZE);
	dev_array<float> d_C(SIZE);

	d_A.set(&h_A[0], SIZE);
	d_B.set(&h_B[0], SIZE);

	start = std::chrono::high_resolution_clock::now();
	matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
	cudaDeviceSynchronize();
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken to multiply a " << N << "x" << N <<  " matrix with gpu: " << duration.count() << " microseconds" << "\n";

	d_C.get(&h_C[0], SIZE);
	cudaDeviceSynchronize();

	float *cpu_C;
	cpu_C = new float[SIZE];

	// cpu matrix multiplication
	start = std::chrono::high_resolution_clock::now();
	float sum;
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			sum = 0.f;
			for (int n = 0; n < N; n++) {
				sum += h_A[row*N + n] * h_B[n*N + col];
			}
			cpu_C[row*N + col] = sum;
		}
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken to multiply a " << N << "x" << N << " matrix with cpu: " << duration.count() << " microseconds" << "\n";

	float max_err = 0.0f;
	double sum_err = 0.0f;
	// error checking
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++) {
			max_err = std::max(abs(cpu_C[ROW * N + COL] - h_C[ROW * N + COL]), max_err);
			sum_err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
		}
	}

	std::cout << "Max Error: " << max_err << "\n";
	std::cout << "Sum Error: " << sum_err << "\n";

	delete[] cpu_C;

	return 0;
}
