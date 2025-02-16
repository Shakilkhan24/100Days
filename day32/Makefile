PROJECT_DIR := $(CURDIR)

COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m
COLOR_RED := \033[31m

HIP_GPU_TARGET := gfx90a

all: build

build: $(PROJECT_DIR)/$(dir)/$(program).out

$(PROJECT_DIR)/$(dir)/$(program).out: $(PROJECT_DIR)/$(dir)/$(program).cpp
	@echo "$(COLOR_YELLOW)Building HIP program $(program) in directory $(dir)...$(COLOR_RESET)"
	@hipcc --offload-arch=$(HIP_GPU_TARGET) -O3 -o $@ $< -lrocblas
	@echo "$(COLOR_GREEN)Build completed for $(program).out in $(dir)$(COLOR_RESET)"

run: $(PROJECT_DIR)/$(dir)/$(program).out
	@echo "$(COLOR_BLUE)Running $(program).out in directory $(dir)...$(COLOR_RESET)"
	@./$(dir)/$(program).out

# Target: Separate rocprof command for kernel profiling
rocprof: $(PROJECT_DIR)/$(dir)/$(program).out
	@echo "$(COLOR_BLUE)Running rocprof for kernel trace on $(program).out in directory $(dir)...$(COLOR_RESET)"
	@mkdir -p $(PROJECT_DIR)/$(dir)/output
	@echo "$(COLOR_GREEN)Kernel profiling completed for $(program).out in $(dir)$(COLOR_RESET)"

# Target: Generate ISA assembly files and kernel resource usage analysis.
isa:
	@echo "$(COLOR_BLUE)Generating ISA and kernel resource usage for $(program) in directory $(dir)...$(COLOR_RESET)"
	@mkdir -p $(PROJECT_DIR)/$(dir)/isa_output
	@hipcc -c --save-temps=obj -O3 -Rpass-analysis=kernel-resource-usage --offload-arch=$(HIP_GPU_TARGET) -o $(PROJECT_DIR)/$(dir)/isa_output/$(program).o $(PROJECT_DIR)/$(dir)/$(program).cpp
	@echo "$(COLOR_GREEN)ISA and resource analysis files saved in $(dir)/isa_output$(COLOR_RESET)"

clean:
	@echo "$(COLOR_RED)Cleaning up .out and ISA files in directory $(dir)...$(COLOR_RESET)"
	@rm -f $(PROJECT_DIR)/$(dir)/*.out
	@rm -rf $(PROJECT_DIR)/$(dir)/isa_output
	@rm -rf $(PROJECT_DIR)/$(dir)/output
	@echo "$(COLOR_GREEN)Clean completed for directory $(dir)$(COLOR_RESET)"

cleanall:
	@echo "$(COLOR_RED)Cleaning up all .out and ISA files in all directories...$(COLOR_RESET)"
	@find $(PROJECT_DIR) -type f -name "*.out" -exec rm -f {} \;
	@find $(PROJECT_DIR) -type d -name "isa_output" -exec rm -rf {} \;
	@find $(PROJECT_DIR) -type d -name "output" -exec rm -rf {} \;
	@echo "$(COLOR_GREEN)Cleanall completed for all directories$(COLOR_RESET)"

help:
	@echo "$(COLOR_BLUE)Usage instructions for HIP Makefile:$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)make dir=<dir> program=<program>$(COLOR_RESET)      		# Build the HIP program <program>.cpp in directory <dir>"
	@echo "$(COLOR_YELLOW)make run dir=<dir> program=<program>$(COLOR_RESET)  		# Run the compiled <program>.out in directory <dir>"
	@echo "$(COLOR_YELLOW)make clean dir=<dir>$(COLOR_RESET)                  		# Clean all .out files in directory <dir>"
	@echo "$(COLOR_YELLOW)make cleanall$(COLOR_RESET)                         		# Clean all .out files in all directories"
	@echo "$(COLOR_YELLOW)make isa dir=<dir> program=<program>$(COLOR_RESET)  		# Generate ISA assembly files and kernel resource usage analysis"
	@echo ""
	@echo "$(COLOR_BLUE)Examples:$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)make dir=matmul_kernels program=kernel_rocblas$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)make run dir=matmul_kernels program=kernel_rocblas$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)make isa dir=matmul_kernels program=kernel_rocblas$(COLOR_RESET)"
