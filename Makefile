TARGET = warp_vote_any.out warp_vote_all.out warp_vote_ballot.out

WARNINGS = -Wall -Wextra
OPT_FLAGS = -O3
# OPT_FLAGS = -O0 -g -DDEBUG

# cuda_profile = yes

# CUDA_HOME=/home/app/cuda/cuda-7.0
CUDA_HOME=/usr/local/cuda

NVCC=$(CUDA_HOME)/bin/nvcc
NVCCFLAGS= $(OPT_FLAGS) -std=c++11 -arch=sm_35 -Xcompiler "$(WARNINGS) $(OPT_FLAGS)" -ccbin=g++
INCLUDE = -I$(CUDA_HOME)/include -I$(CUDA_HOME)/samples/common/inc
ifeq ($(cuda_profile), yes)
NVCCFLAGS += -lineinfo -Xptxas -v
endif

all: $(TARGET)

warp_vote_any.out: warp_vote.cu
	$(NVCC) -DTEST_ANY $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

warp_vote_all.out: warp_vote.cu
	$(NVCC) -DTEST_ALL $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

warp_vote_ballot.out: warp_vote.cu
	$(NVCC) -DTEST_BALLOT $(NVCCFLAGS) $(INCLUDE) $< $(LIBRARY) -o $@

clean:
	rm -f $(TARGET) *~ *.core
