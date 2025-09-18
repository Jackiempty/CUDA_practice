NVCC := nvcc
BUILD := build


.PHONY: all naive optimized

all: naive.cuh optimized.cuh
	@mkdir -p $(BUILD)
	@$(NVCC) main.cu -o ./build/main

clean:
	rm -rf $(BUILD)