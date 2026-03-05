# High-Performance DFO/uDFO Makefile
#
# Targets:
#   all       - Build all targets
#   dfo_gpu   - Build GPU DFO executable
#   clean     - Remove build artifacts
#   test      - Run basic tests
#   benchmark - Run performance benchmark

# CUDA compiler
NVCC = nvcc

# Detect CUDA architecture automatically from the first available device
# Falls back to sm_61 (Pascal) if detection fails
CUDA_ARCH ?= $(shell python3 -c "import subprocess,re; out=subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],capture_output=True,text=True).stdout.strip().split('\n')[0].replace('.',''); print('sm_'+out)" 2>/dev/null || echo sm_61)

# Compiler flags
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++17
NVCC_FLAGS += -Xcompiler -Wall -Xcompiler -Wextra
NVCC_FLAGS += --ftz=false  # Keep IEEE 754 denormal behavior (double precision correctness)
NVCC_FLAGS += -lineinfo  # For profiling

# Debug flags (use: make DEBUG=1)
ifdef DEBUG
    NVCC_FLAGS = -g -G -arch=$(CUDA_ARCH) -std=c++17 -DDEBUG
endif

# Include paths
INCLUDES = -I./include

# Linker flags
LDFLAGS = -lcurand

# Source files
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Source and header files
CUDA_SRCS = $(SRC_DIR)/dfo_gpu.cu
HEADERS = $(wildcard $(INCLUDE_DIR)/*.cuh)

# Targets
TARGET = dfo_gpu

# Default target
all: $(BUILD_DIR) $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build main executable
$(TARGET): $(CUDA_SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(CUDA_SRCS) $(LDFLAGS)

# Clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Run basic test
test: $(TARGET)
	@echo "=== Running Standard DFO Test ==="
	./$(TARGET) standard sphere 100 30 500
	@echo ""
	@echo "=== Running uDFO-1500p Test ==="
	./$(TARGET) udfo1500 sphere 100 30 500
	@echo ""
	@echo "=== Running uDFO-z5 Test ==="
	./$(TARGET) udfoz5 rastrigin 100 30 500

# Run performance benchmark
benchmark: $(TARGET)
	./$(TARGET) benchmark 1000 100 5000

# High-dimensional test
test_highdim: $(TARGET)
	@echo "=== Testing High Dimensions (D=500) ==="
	./$(TARGET) udfo1500 sphere 200 500 2000

# Large population test
test_largepop: $(TARGET)
	@echo "=== Testing Large Population (N=10000) ==="
	./$(TARGET) udfo1500 sphere 10000 30 500

# Profile with nvprof (if available)
profile: $(TARGET)
	nvprof --print-gpu-trace ./$(TARGET) udfo1500 sphere 1000 100 1000

# Show CUDA device info
deviceinfo:
	nvidia-smi

# Help
help:
	@echo "High-Performance GPU DFO/uDFO"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all targets (default)"
	@echo "  clean       - Remove build artifacts"
	@echo "  test        - Run basic tests"
	@echo "  benchmark   - Run performance benchmark"
	@echo "  test_highdim - Test with high dimensions"
	@echo "  test_largepop - Test with large population"
	@echo "  profile     - Profile with nvprof"
	@echo ""
	@echo "Options:"
	@echo "  CUDA_ARCH=sm_XX  - Set CUDA architecture (default: sm_75)"
	@echo "  DEBUG=1          - Build with debug symbols"
	@echo ""
	@echo "Example:"
	@echo "  make CUDA_ARCH=sm_86"
	@echo "  ./dfo_gpu udfo1500 sphere 1000 100 5000"

.PHONY: all clean test benchmark test_highdim test_largepop profile deviceinfo help
