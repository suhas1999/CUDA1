#include <chrono>
#include <cstdlib>
#include <iostream>

__global__ void vecAdd(float *A, float *B, float *C, long long N) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = blockDim.x * gridDim.x;

  for (long long i = idx; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <K (in millions)> <Scenario (1, 2, or 3)>\n";
    return 1;
  }

  int K = std::atoi(argv[1]);
  int scenario = std::atoi(argv[2]);

  if (K <= 0) {
    std::cerr << "K must be a positive integer.\n";
    return 1;
  }

  long long N = static_cast<long long>(K) * 1000000;
  size_t size = N * sizeof(float);

  // 1. Allocate memory on host
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
    std::cerr << "Host memory allocation failed!\n";
    free(h_A);
    free(h_B);
    free(h_C);
    return 1;
  }

  // Initialize arrays A and B on host
  for (long long i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = static_cast<float>(N - i);
  }

  // 2. Allocate memory on GPU
  float *d_A, *d_B, *d_C;
  if (cudaMalloc(&d_A, size) != cudaSuccess ||
      cudaMalloc(&d_B, size) != cudaSuccess ||
      cudaMalloc(&d_C, size) != cudaSuccess) {
    std::cerr << "Device memory allocation failed!\n";
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 1;
  }

  // Configure Grid and Block dimensions based on Scenario
  int threads;
  long long blocks;

  if (scenario == 1) {
    threads = 1;
    blocks = 1;
  } else if (scenario == 2) {
    threads = 256;
    blocks = 1;
  } else if (scenario == 3) {
    threads = 256;
    blocks = (N + threads - 1) / threads;
  } else {
    std::cerr << "Invalid scenario. Must be 1, 2, or 3.\n";
    return 1;
  }

  // Warm-up the CUDA Context (Forces initialization overhead to happen now)
  cudaFree(0);

  // CPU Timer (Tracks Explicit App-Level Time: Memcpy + Kernel + Memcpy)
  auto start_cpu = std::chrono::high_resolution_clock::now();

  // 3. Copy data to GPUs
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // Create CUDA events for Profiling STRICTLY the Kernel performance
  cudaEvent_t start_kernel, stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);

  cudaEventRecord(start_kernel);

  // 4. Launch the Kernel
  vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);

  float kernel_ms = 0;
  cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);

  // 5. Copy data back from GPUs
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // Stop CPU Timer
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_cpu - start_cpu;

  // Print profiling results
  std::cout << "Explicit Memory - K = " << K << "M | Scenario = " << scenario;
  std::cout << " | Configuration: <<<" << blocks << ", " << threads << ">>>\n";
  std::cout << "STRICT Kernel Execution Time: " << kernel_ms / 1000.0 << " seconds (" << kernel_ms << " ms)\n";
  std::cout << "TOTAL Round-Trip Time (Memcpy + Kernel + Memcpy): " << total_duration.count() << " seconds\n";

  // 6. Free the memory on the device and host
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
