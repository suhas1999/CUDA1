###
### Makefile for Part-C: CUDA Convolution Exercises
### C1: Naive convolution kernel
### C2: Tiled convolution kernel with shared memory
### C3: cuDNN convolution
###

SDK_INSTALL_PATH := /usr/local/cuda
NVCC             := $(SDK_INSTALL_PATH)/bin/nvcc
LIB              := -L$(SDK_INSTALL_PATH)/lib64
OPTIONS          := -O3

# cuDNN link flag (used only for C3)
CUDNN_LIB        := -lcudnn

EXECS := c1 c2 c3

all: $(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
# C1: Naive convolution kernel
#######################################################################
c1: c1.cu
	$(NVCC) $< -o $@ $(LIB) $(OPTIONS)

#######################################################################
# C2: Tiled convolution with shared memory
# Use -DBLOCK_X=N -DBLOCK_Y=M to override block size at compile time
#######################################################################
c2: c2.cu
	$(NVCC) $< -o $@ $(LIB) $(OPTIONS)

#######################################################################
# C3: cuDNN convolution (requires -lcudnn)
#######################################################################
c3: c3.cu
	$(NVCC) $< -o $@ $(LIB) $(CUDNN_LIB) $(OPTIONS)
