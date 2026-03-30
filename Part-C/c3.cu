#include <cudnn.h>
#include <iomanip>
#include <iostream>

// Dimensions (same as c1.cu and c2.cu)
const int H = 1024;
const int W = 1024;
const int C = 3;
const int FW = 3;
const int FH = 3;
const int K = 64;
const int P = 1;

// Helper macros for flattened NCHW / KCRS indexing (Standard Deep Learning
// Format) I dims: Batch(1), C, H, W (Row-Major: y is row, x is col)
#define I_IDX(c, x, y) ((c) * H * W + (y) * W + (x))

// F dims: K, C, FH, FW (Row-Major: y is row, x is col)
#define F_IDX(k, c, x, y) ((k) * C * FH * FW + (c) * FH * FW + (y) * FW + (x))

// O dims: Batch(1), K, H, W
#define O_IDX(k, x, y) ((k) * H * W + (y) * W + (x))

// Macro to catch cuDNN errors
#define checkCUDNN(expression)                                                 \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << cudnnGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

int main() {
  // 1. Allocate host matrices
  // Notice we NO LONGER need I0 (the padded array) explicitly.
  // cuDNN handles padding logically via the ConvolutionDescriptor!
  size_t size_I = C * W * H * sizeof(double);
  size_t size_F = K * C * FW * FH * sizeof(double);
  size_t size_O = K * W * H * sizeof(double);

  double *h_I = (double *)malloc(size_I);
  double *h_F = (double *)malloc(size_F);
  double *h_O = (double *)malloc(size_O);

  // Initialize I (Input) using identical logic to q1/q2
  for (int c = 0; c < C; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        h_I[I_IDX(c, x, y)] = (double)(c * (x + y));
      }
    }
  }

  // Initialize F (Filter/Weights) using identical logic to q1/q2
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < FH; ++y) {
        for (int x = 0; x < FW; ++x) {
          h_F[F_IDX(k, c, x, y)] = (double)((c + k) * (x + y));
        }
      }
    }
  }

  // 2. Allocate device matrices
  double *d_I, *d_F, *d_O;
  cudaMalloc(&d_I, size_I);
  cudaMalloc(&d_F, size_F);
  cudaMalloc(&d_O, size_O);

  // Copy data to device
  cudaMemcpy(d_I, h_I, size_I, cudaMemcpyHostToDevice);
  cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice);

  // -------------------------------------------------------------
  // 3. cuDNN Setup Phase
  // -------------------------------------------------------------

  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  // --- Input Descriptor (xDesc) ---
  cudnnTensorDescriptor_t xDesc;
  checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
  // Dimensions map to: N (Batch), C (Channels), H (Height), W (Width)
  checkCUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_DOUBLE, 1, C, H, W));

  // --- Filter Descriptor (wDesc) ---
  cudnnFilterDescriptor_t wDesc;
  checkCUDNN(cudnnCreateFilterDescriptor(&wDesc));
  // Dimensions map to: K (Output Channels), C (Input Channels), FH (Filter
  // Height), FW (Filter Width)
  checkCUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_DOUBLE,
                                        CUDNN_TENSOR_NCHW, K, C, FH, FW));

  // --- Convolution Descriptor (convDesc) ---
  cudnnConvolutionDescriptor_t convDesc;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
  // Pad_h = P, Pad_w = P, Stride_h = 1, Stride_w = 1, Dilation_h = 1,
  // Dilation_w = 1 CUDNN_CONVOLUTION performs kernel flipping (true
  // mathematical convolution), matching the flip in q1: F[FH-1-j][FW-1-i]. This
  // ensures the checksum matches.
  checkCUDNN(cudnnSetConvolution2dDescriptor(
      convDesc, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

  // --- Output Descriptor (yDesc) ---
  cudnnTensorDescriptor_t yDesc;
  checkCUDNN(cudnnCreateTensorDescriptor(&yDesc));

  // We can ask cuDNN what the output dimension will be mathematically
  int out_n, out_c, out_h, out_w;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      convDesc, xDesc, wDesc, &out_n, &out_c, &out_h, &out_w));
  // Create output descriptor using these calculated dimensions
  checkCUDNN(cudnnSetTensor4dDescriptor(
      yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, out_n, out_c, out_h, out_w));

  // -------------------------------------------------------------
  // 4. Algorithm Selection (Using v7)
  // -------------------------------------------------------------
  int requestedAlgoCount = 1; // We only want the best 1 algorithm
  int returnedAlgoCount = 0;  // How many algorithms cuDNN actually returned
  cudnnConvolutionFwdAlgoPerf_t perfResults;

  // Ask cuDNN to search the heuristic database for the absolute best algorithm
  // for our tensor sizes.
  checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
      cudnn, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount,
      &returnedAlgoCount, &perfResults));

  // the chosen algorithm is located inside perfResults.algo
  cudnnConvolutionFwdAlgo_t algo = perfResults.algo;

  // -------------------------------------------------------------
  // 5. Workspace Allocation
  // -------------------------------------------------------------
  size_t workspace_size = 0;
  // Let cuDNN tell us exactly how much scratch-pad memory it needs for the
  // chosen algorithm
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, xDesc, wDesc, convDesc, yDesc, algo, &workspace_size));

  void *d_workspace = nullptr;
  if (workspace_size > 0) {
    cudaMalloc(&d_workspace, workspace_size);
  }

  // -------------------------------------------------------------
  // 6. Execution & Profiling (CUDA Events for accurate GPU-only timing)
  // -------------------------------------------------------------
  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);

  double alpha = 1.0;
  double beta = 0.0; // Overwrite output (Output = Alpha * Conv + Beta * Output)

  cudaEventRecord(ev_start);
  checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, xDesc, d_I, wDesc, d_F,
                                     convDesc, algo, d_workspace,
                                     workspace_size, &beta, yDesc, d_O));
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float kernel_ms = 0.0f;
  cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  // -------------------------------------------------------------
  // 7. Verification
  // -------------------------------------------------------------

  // Copy output back to host
  cudaMemcpy(h_O, d_O, size_O, cudaMemcpyDeviceToHost);

  // Compute checksum
  double checksum = 0.0;
  for (int k = 0; k < K; ++k) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        checksum += h_O[O_IDX(k, x, y)];
      }
    }
  }

  // Report checksum and time (3 decimal places)
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Checksum: " << checksum << "\n";
  std::cout << "cuDNN Kernel Execution Time: " << kernel_ms << " ms\n";

  // Cleanup
  if (d_workspace != nullptr)
    cudaFree(d_workspace);
  cudaFree(d_I);
  cudaFree(d_F);
  cudaFree(d_O);

  checkCUDNN(cudnnDestroyTensorDescriptor(xDesc));
  checkCUDNN(cudnnDestroyFilterDescriptor(wDesc));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(yDesc));
  checkCUDNN(cudnnDestroy(cudnn));

  free(h_I);
  free(h_F);
  free(h_O);

  return 0;
}
