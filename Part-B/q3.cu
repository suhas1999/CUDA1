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

  // 1. Allocate Unified Memory accessible from BOTH CPU and GPU
  float *A, *B, *C;
  if (cudaMallocManaged(&A, size) != cudaSuccess ||
      cudaMallocManaged(&B, size) != cudaSuccess ||
      cudaMallocManaged(&C, size) != cudaSuccess) {
    std::cerr << "Unified Memory allocation failed!\n";
    return 1;
  }

  // 2. Initialize arrays directly via CPU using the Unified Pointers
  // This physically allocates the memory pages on the CPU RAM first (First
  // Touch)
  for (long long i = 0; i < N; ++i) {
    A[i] = static_cast<float>(i);
    B[i] = static_cast<float>(N - i);
  }

  // Configure Grid and Block dimensions
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

  // Warm-up the CUDA Context (Initialization overhead only, no kernel
  // execution)
  cudaFree(0);

  // CPU Timer (Tracks Explicit App-Level Time: Implicit Page Faults + Kernel +
  // Verification Page Faults)
  auto start_cpu = std::chrono::high_resolution_clock::now();

  // Create CUDA events for Profiling STRICTLY the Kernel performance + GPU Page
  // Faults
  cudaEvent_t start_kernel, stop_kernel;
  cudaEventCreate(&start_kernel);
  cudaEventCreate(&stop_kernel);

  cudaEventRecord(start_kernel);

  // 3. Launch the Kernel
  // *NOTE*: Because we did not do a warmup run, this kernel execution will
  // trigger massive Page Faults causing the GPU to stall while data migrates
  // over PCIe.
  vecAdd<<<blocks, threads>>>(A, B, C, N);

  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);

  float kernel_ms = 0;
  cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);

  // 4. Force CPU to pull the data back from the GPU!
  // In `q2.cu`, you explicitly called cudaMemcpyDeviceToHost at this exact
  // step. In Unified Memory, the data natively stays on the GPU until the CPU
  // tries to read it. We do a dummy read here to force the Page Faults back to
  // the CPU to keep the timer fair!
  volatile float dummy = 0;
  for (long long i = 0; i < N; i += 4096 / sizeof(float)) {
    // Read one float per 4KB memory page to trigger the migration
    dummy += C[i];
  }

  // Stop total CPU Timer
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_cpu - start_cpu;

  // Print profiling results
  std::cout << "Unified Memory - K = " << K << "M | Scenario = " << scenario;
  std::cout << " | Configuration: <<<" << blocks << ", " << threads << ">>>\n";
  std::cout
      << "Kernel execution (INCLUDES Host-to-Device Page Fault overhead): "
      << kernel_ms / 1000.0 << " seconds (" << kernel_ms << " ms)\n";
  std::cout << "TOTAL Round-Trip Time (H2D Faults + Kernel + D2H Faults): "
            << total_duration.count() << " seconds\n";

  // 5. Free Unified Memory
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
