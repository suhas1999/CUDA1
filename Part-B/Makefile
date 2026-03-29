###
### Makefile for Part-B CUDA assignment
###

SDK_INSTALL_PATH := /usr/local/cuda
NVCC := $(SDK_INSTALL_PATH)/bin/nvcc
CXX := g++

# Compiler options
OPTIONS := -O3
CXX_OPTIONS := -O3 -std=c++11

EXECS := q1 q2 q3

all: $(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################

# Compile CPU-only q1
q1: q1.cpp
	$(CXX) $< -o $@ $(CXX_OPTIONS)

# Compile CUDA q2 (Explicit Memory)
q2: q2.cu
	$(NVCC) $< -o $@ $(OPTIONS)

# Compile CUDA q3 (Unified Memory)
q3: q3.cu
	$(NVCC) $< -o $@ $(OPTIONS)

