#include <cstring>
#include <iomanip>
#include <iostream>

// Dimensions
const int H = 1024;
const int W = 1024;
const int C = 3;
const int FW = 3;
const int FH = 3;
const int K = 64;
const int P = 1;

// Configurable block dimensions (can be overridden via compiler flags)
#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif
#ifndef BLOCK_Z
#define BLOCK_Z 1
#endif

// Helper macros for flattened indexing
// These map the multi-dimensional indices to a flat 1D array index.
// I  dims: C, H, W (Row-Major: y is row, x is col)
#define I_IDX(c, x, y) ((c) * H * W + (y) * W + (x))
// I0 dims: C, H+2P, W+2P
#define I0_IDX(c, x, y)                                                        \
  ((c) * (H + 2 * P) * (W + 2 * P) + (y) * (W + 2 * P) + (x))
// F  dims: K, C, FH, FW (Row-Major: j is row, i is col)
#define F_IDX(k, c, i, j) ((k) * C * FH * FW + (c) * FH * FW + (j) * FW + (i))
// O  dims: K, H, W
#define O_IDX(k, x, y) ((k) * H * W + (y) * W + (x))

// CUDA kernel for convolution
__global__ void conv_kernel(const double *__restrict__ I0,
                            const double *__restrict__ F,
                            double *__restrict__ O) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < W && y < H && k < K) {
    double sum = 0.0;
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < FW; ++i) {
        for (int j = 0; j < FH; ++j) {
          sum += F[F_IDX(k, c, FW - 1 - i, FH - 1 - j)] *
                 I0[I0_IDX(c, x + i, y + j)];
        }
      }
    }
    O[O_IDX(k, x, y)] = sum;
  }
}

int main() {
  // 1. Allocate and initialize host matrices
  size_t size_I = C * W * H * sizeof(double);
  size_t size_I0 = C * (W + 2 * P) * (H + 2 * P) * sizeof(double);
  size_t size_F = K * C * FW * FH * sizeof(double);
  size_t size_O = K * W * H * sizeof(double);

  double *h_I = (double *)malloc(size_I);
  double *h_I0 = (double *)malloc(size_I0);
  double *h_F = (double *)malloc(size_F);
  double *h_O = (double *)malloc(size_O);

  // Initialize I[c, x, y] = c * (x + y)
  for (int c = 0; c < C; ++c) {
    for (int x = 0; x < W; ++x) {
      for (int y = 0; y < H; ++y) {
        h_I[I_IDX(c, x, y)] = (double)(c * (x + y));
      }
    }
  }

  // Initialize F[k, c, i, j] = (c + k) * (i + j)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < FW; ++i) {
        for (int j = 0; j < FH; ++j) {
          h_F[F_IDX(k, c, i, j)] = (double)((c + k) * (i + j));
        }
      }
    }
  }

  // Initialize padding tensor I0 to zeros, and copy I into it
  memset(h_I0, 0, size_I0);
  for (int c = 0; c < C; ++c) {
    for (int x = 0; x < W; ++x) {
      for (int y = 0; y < H; ++y) {
        h_I0[I0_IDX(c, x + P, y + P)] = h_I[I_IDX(c, x, y)];
      }
    }
  }

  // 2. Allocate device memory
  double *d_I0, *d_F, *d_O;
  cudaMalloc(&d_I0, size_I0);
  cudaMalloc(&d_F, size_F);
  cudaMalloc(&d_O, size_O);

  // 3. Copy data to device
  cudaMemcpy(d_I0, h_I0, size_I0, cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice);

  // 4. Set up grid and block dimensions for the kernel
  // We map threads to output elements O[k, x, y]
  dim3 threadsPerBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 numBlocks((W + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (H + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 (K + threadsPerBlock.z - 1) / threadsPerBlock.z);

  // 5. Create CUDA events for accurate kernel-only timing
  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  // 6. Launch the kernel (GPU hardware timestamps the start and end)
  cudaEventRecord(ev_start);
  conv_kernel<<<numBlocks, threadsPerBlock>>>(d_I0, d_F, d_O);
  cudaEventRecord(ev_stop);

  // Wait until the stop event is recorded before asking for elapsed time
  cudaEventSynchronize(ev_stop);

  float kernel_ms = 0.0f;
  cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  // 7. Copy output back to host
  cudaMemcpy(h_O, d_O, size_O, cudaMemcpyDeviceToHost);

  // 8. Compute checksum (sum along all dimensions)
  double checksum = 0.0;
  for (int k = 0; k < K; ++k) {
    for (int x = 0; x < W; ++x) {
      for (int y = 0; y < H; ++y) {
        checksum += h_O[O_IDX(k, x, y)];
      }
    }
  }

  // 9. Report checksum and time (3 decimal places)
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Checksum: " << checksum << "\n";
  std::cout << "Kernel Execution Time: " << kernel_ms << " ms\n";

  // 10. Free memory
  cudaFree(d_I0);
  cudaFree(d_F);
  cudaFree(d_O);

  free(h_I);
  free(h_I0);
  free(h_F);
  free(h_O);

  return 0;
}
