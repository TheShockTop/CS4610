#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#include <string>
#include <iostream>
#include <chrono>


__global__ void add(int n, float *x, float *y)
{
	for (int i = 0; i < n; i++) {
		y[i] = x[i] + y[i];
	}
}

void cpu_add(int n, float *x, float *y)
{
	for (int i = 0; i < n; i++) {
		y[i] = x[i] + y[i];
	}
}

void check_error(float *x, float *y, int N) {
	float maxError = 0.0f;

	for (int i = 0; i < N; i++) {
		maxError = max(maxError, abs(y[i] - 3.0f));
	}

	printf("Max error: %f\n", maxError);
}

void set_array(float *x, float *y, int N) {
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
}

void gpu_function(int max_threads) {
	int N = 1 << 20;
	float *x, *y, *d_x, *d_y;

	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));

	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	set_array(x, y, N);

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	add<<<1, max_threads>>>(N, d_x, d_y);

	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	// cpu bound function
	check_error(x, y, N);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}

void cpu_function() {
	int N = 1 << 20; // 1M elements

	float *x = new float[N];
	float *y = new float[N];

	// initialize x and y arrays on the host
	set_array(x, y, N);

	// Run kernel on 1M elements on the CPU
	cpu_add(N, x, y);

	// Check for errors (all values should be 3.0f)
	check_error(x, y, N);

	// Free memory
	delete[] x;
	delete[] y;
}

int main(void)
{
	auto start = std::chrono::high_resolution_clock::now();
	cpu_function();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by cpu_function: " << duration.count() << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	gpu_function(1);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by gpu_function with 1 thread: " << duration.count() << " microseconds" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	gpu_function(256);
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by gpu_function with 256 threads: " << duration.count() << " microseconds" << std::endl;

	std::string timeout;
	std::cin >> timeout;

	return 0;
}
