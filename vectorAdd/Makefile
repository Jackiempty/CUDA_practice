NVCC := nvcc
BUILD := build


.PHONY: all naive optimized

all: naive.cuh optimized.cuh
	@$(NVCC) main.cu -o main

clean:
	rm -rf $(BUILD)