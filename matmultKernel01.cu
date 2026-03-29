///
/// matmultKernel00.cu
/// For COMS E6998 Spring 2023
/// Instructor: Parajit Dube and Kaoutar El Maghraoui
/// Based on code from the CUDA Programming Guide
/// Modified by Wim Bohm and David Newman
/// Created: 2011-01-27
/// Last Modified: 2011-02-23 DVN
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Copy this file and modify the MatMultKernel device function for
/// each of your experiments.
///

#include "matmultKernel.h"

// Define a gpu kernel to perform matrix multiplication
// of A x B = C.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_row = threadIdx.y;
  int thread_col = threadIdx.x;
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row +
                     FOOTPRINT_SIZE * block_col];

  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue1_1 = 0;
  float Cvalue1_2 = 0;
  float Cvalue2_1 = 0;
  float Cvalue2_2 = 0;

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results

  for (int m = 0; m < (A.width / FOOTPRINT_SIZE); ++m) {

    // Get Asub and Bsub descriptors
    Asub =
        &A.elements[A.stride * FOOTPRINT_SIZE * block_row + FOOTPRINT_SIZE * m];
    Bsub =
        &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col];

    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

#pragma unroll
    for (int i = 0; i < FOOTPRINT_SIZE / BLOCK_SIZE; ++i) {

#pragma unroll
      for (int j = 0; j < FOOTPRINT_SIZE / BLOCK_SIZE; ++j) {

        shared_A[i * BLOCK_SIZE + thread_row][j * BLOCK_SIZE + thread_col] =
            Asub[thread_row * A.stride + i * BLOCK_SIZE * A.stride +
                 j * BLOCK_SIZE + thread_col];
        shared_B[i * BLOCK_SIZE + thread_row][j * BLOCK_SIZE + thread_col] =
            Bsub[thread_row * B.stride + i * BLOCK_SIZE * B.stride +
                 j * BLOCK_SIZE + thread_col];
      }
    }

    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll

    for (int e = 0; e < FOOTPRINT_SIZE; ++e) {

      float a0 = shared_A[thread_row][e];
      float a1 = shared_A[thread_row + BLOCK_SIZE][e];
      float b0 = shared_B[e][thread_col];
      float b1 = shared_B[e][thread_col + BLOCK_SIZE];
      Cvalue1_1 += a0 * b0;
      Cvalue1_2 += a0 * b1;
      Cvalue2_1 += a1 * b0;
      Cvalue2_2 += a1 * b1;
    }

    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
  Csub[thread_row * C.stride + thread_col] = Cvalue1_1;
  Csub[(thread_row + BLOCK_SIZE) * C.stride + thread_col] = Cvalue2_1;
  Csub[thread_row * C.stride + (thread_col + BLOCK_SIZE)] = Cvalue1_2;
  Csub[(thread_row + BLOCK_SIZE) * C.stride + (thread_col + BLOCK_SIZE)] =
      Cvalue2_2;
}
