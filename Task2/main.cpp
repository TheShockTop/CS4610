#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#include <string>
#include <iostream>
#include <chrono>

__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = a * x[i] + y[i];
}

__global__
void gpu_add(int n, const float *x, const float *y, float *z) {
	/*
	for (int i = 0; i < n; i++) {
	z[i] = x[i] + y[i];
	}
	*/

	int i = blockIdx.x*blockDim.x + threadIdx.x;;
	if (i < n) z[i] = x[i] + y[i];
}

__global__
void gpu_multiply(const int n, const float **x, const float **y, float **z) {

}

__global__
void gpu_add_parallel(const int n, const float *x, const float *y, float *z) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride) {
		z[i] = x[i] + y[i];
	}
}

void cpu_add(const int n, const float *x, const float *y, float *z) {
	for (int i = 0; i < n; i++) {
		z[i] = x[i] + y[i];
	}
}

void cpu_multiply(const int n, const float **x, const float **y, float **z) {

}

void cpu_function() {
	std::chrono::steady_clock::time_point start, stop, stop_check, start_check;
	std::chrono::microseconds duration;

	int n = 1 << 20;

	float *x = new float[n];
	float *y = new float[n];
	float *z = new float[n];

	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	start = std::chrono::high_resolution_clock::now();
	cpu_add(n, x, y, z);
	stop = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken to add with cpu: " << duration.count() << " microseconds\n";

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	start_check = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n; i++) {
		maxError = fmax(maxError, fabs(z[i] - 3.0f));
	}
	stop_check = std::chrono::high_resolution_clock::now();

	std::cout << "Max error: " << maxError << "\n";

	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_check - start_check);
	std::cout << "Time taken to check with cpu: " << duration.count() << " microseconds\n";
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_check - start);
	std::cout << "Total time elapsed with cpu: " << duration.count() << " microseconds\n\n";

	delete[] x;
	delete[] y;
	delete[] z;
}

void gpu_function() {
	std::chrono::steady_clock::time_point start, stop, start_check, stop_check;
	std::chrono::microseconds duration;

	int threads = 256;

	int n = 1 << 20;

	float *x, *y, *z, *gpu_x, *gpu_y, *gpu_z;

	x = new float[n];
	y = new float[n];
	z = new float[n];

	/*
	cudaMallocManaged(&x, n * sizeof(float));
	cudaMallocManaged(&y, n * sizeof(float));
	cudaMallocManaged(&z, n * sizeof(float));
	*/

	cudaMalloc(&gpu_x, n * sizeof(float));
	cudaMalloc(&gpu_y, n * sizeof(float));
	cudaMalloc(&gpu_z, n * sizeof(float));


	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
		z[i] = 0.0f;
	}

	cudaMemcpy(gpu_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_z, z, n * sizeof(float), cudaMemcpyHostToDevice);

	start = std::chrono::high_resolution_clock::now();
	gpu_add << <(n + 255) / 256, 256 >> > (n, gpu_x, gpu_y, gpu_z);
	cudaMemcpy(z, gpu_z, n * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken to add with gpu of " << threads << " threads: " << duration.count() << " microseconds" << "\n";

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	start_check = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n; i++) {
		maxError = fmax(maxError, fabs(z[i] - 3.0f));
	}
	stop_check = std::chrono::high_resolution_clock::now();

	std::cout << "Max error: " << maxError << "\n";

	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_check - start_check);
	std::cout << "Time taken to check with cpu: " << duration.count() << " microseconds\n";
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_check - start);
	std::cout << "Total time elapsed with gpu of " << threads << " threads: " << duration.count() << " microseconds\n\n";

	cudaFree(gpu_x);
	cudaFree(gpu_y);
	cudaFree(gpu_z);
	delete[] x;
	delete[] y;
	delete[] z;
}

void gpu_parallel() {
	std::chrono::steady_clock::time_point start, stop, start_check, stop_check;
	std::chrono::microseconds duration;

	int n = 1 << 20;

	int blockSize = 256;
	int numBlocks = (n + blockSize - 1) / blockSize;

	float *x, *y, *z, *gpu_x, *gpu_y, *gpu_z;

	x = new float[n];
	y = new float[n];
	z = new float[n];

	cudaMalloc(&gpu_x, n * sizeof(float));
	cudaMalloc(&gpu_y, n * sizeof(float));
	cudaMalloc(&gpu_z, n * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < n; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
		z[i] = 0.0f;
	}

	cudaMemcpy(gpu_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_z, z, n * sizeof(float), cudaMemcpyHostToDevice);

	start = std::chrono::high_resolution_clock::now();
	gpu_add_parallel << <numBlocks, blockSize >> > (n, x, y, z);
	cudaMemcpy(z, gpu_z, n * sizeof(float), cudaMemcpyDeviceToHost);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken to add with parallelized gpu: " << duration.count() << " microseconds" << "\n";

	// Check for errors (all values should be 4.0f)
	float maxError = 0.0f;
	start_check = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < n; i++) {
		maxError = fmax(maxError, fabs(z[i] - 4.0f));
	}
	stop_check = std::chrono::high_resolution_clock::now();

	std::cout << "Max error: " << maxError << "\n";

	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_check - start_check);
	std::cout << "Time taken to check with cpu: " << duration.count() << " microseconds\n";
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_check - start);
	std::cout << "Total time elapsed with parallelized gpu: " << duration.count() << " microseconds\n\n";

	cudaFree(gpu_x);
	cudaFree(gpu_y);
	cudaFree(gpu_z);
	delete[] x;
	delete[] y;
	delete[] z;
}

void tutorial() {
	int N = 1 << 20;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));

	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	saxpy << <(N + 255) / 256, 256 >> > (N, 2.0f, d_x, d_y);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i] - 4.0f));
	printf("Max error: %f\n", maxError);

	for (int i = 0; i < N; i++) {
		std::cout << y[i] << " ";
	}

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}

int main(void)
{
	std::string string_test;

	cpu_function();
	//tutorial();
	gpu_function();
	gpu_parallel();

	std::cin >> string_test;

	return 0;
}
