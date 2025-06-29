This project implements four computationally intensive image processing algorithms:
1. **Gaussian Blur** - Smoothing filter using convolution
2. **Edge Detection** - Sobel operator for edge detection  
3. **Histogram Equalization** - Contrast enhancement
4. **Otsu Thresholding** - Automatic optimal threshold selection for binarization# Parallel Image Processing System

A comprehensive implementation of parallel image processing algorithms using OpenMP and MPI, with automatic benchmarking and performance visualization.

## 🎯 Project Overview

This project implements three computationally intensive image processing algorithms:
1. **Gaussian Blur** - Smoothing filter using convolution
2. **Edge Detection** - Sobel operator for edge detection  
3. **Histogram Equalization** - Contrast enhancement
4. **Otsu Thresholding** - automatic image thresholding

Each algorithm is implemented in three versions:
- **Sequential** - Single-threaded baseline
- **OpenMP** - Shared-memory parallelization
- **MPI** - Distributed-memory parallelization

## 📋 Requirements

### System Requirements
- Linux/Unix system (tested on Ubuntu/CentOS)
- C++ compiler with C++17 support
- OpenMP support
- MPI implementation (OpenMPI or MPICH)
- Python 3.6+ with pip

### Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential libopenmpi-dev openmpi-bin python3 python3-pip

# CentOS/RHEL
sudo yum install gcc-c++ openmpi openmpi-devel python3 python3-pip
module load mpi/openmpi-x86_64  # May be needed on some systems

# Python packages (auto-installed by run script)
pip3 install pandas matplotlib seaborn numpy
```

## 🚀 Quick Start

1. **Clone/Download the files** and place them in a directory
2. **Make the run script executable:**
   ```bash
   chmod +x run_benchmark.sh
   ```
3. **Run the complete benchmark:**
   ```bash
   ./run_benchmark.sh
   ```

This will automatically:
- Build the parallel image processor
- Run benchmarks on multiple image sizes (512x512, 1024x1024, 2048x2048)
- Generate performance graphs
- Create detailed analysis reports

## 📁 Project Structure

```
parallel-image-processing/
├── parallel_image_processor.cpp  # Main implementation
├── Makefile                      # Build configuration
├── run_benchmark.sh             # Complete benchmark script
├── plot_speedup.py             # Visualization generator
├── analyze_performance.py      # Performance analysis
├── README.md                   # This documentation
└── Generated files:
    ├── parallel_image_processor # Compiled executable
    ├── speedup_results.csv     # Raw benchmark data
    ├── speedup_analysis.png    # Performance graphs
    └── performance_summary.csv # Analysis summary
```

## 🔧 Manual Build and Run

### Building
```bash
# Using Makefile
make clean && make

# Manual compilation
mpicxx -std=c++17 -O3 -fopenmp -o parallel_image_processor parallel_image_processor.cpp -lm
```

### Running Individual Tests
```bash
# Sequential only (1 thread)
export OMP_NUM_THREADS=1
./parallel_image_processor

# OpenMP with 4 threads
export OMP_NUM_THREADS=4
./parallel_image_processor

# MPI with 4 processes
mpirun -np 4 ./parallel_image_processor

# Combined OpenMP + MPI
export OMP_NUM_THREADS=2
mpirun -np 2 ./parallel_image_processor
```

## 📊 Performance Analysis

### Generated Outputs

1. **speedup_results.csv** - Raw performance data
2. **speedup_analysis.png** - Comprehensive performance graphs showing:
   - Speedup comparison by algorithm
   - Speedup vs image size scaling
   - Execution time comparisons
   - Parallel efficiency analysis

3. **performance_summary.csv** - Algorithm-specific performance summary

### Understanding the Results

The system measures and reports:
- **Speedup**: Sequential time ÷ Parallel time
- **Efficiency**: (Speedup ÷ Number of cores) × 100%
- **Scalability**: Performance consistency across different image sizes

### Expected Performance Characteristics

| Algorithm | OpenMP Expected | MPI Expected | Notes |
|-----------|----------------|--------------|--------|
| Gaussian Blur | 2.5-3.5x | 2.0-3.0x | Computation-heavy, good parallelization |
| Edge Detection | 3.0-4.0x | 2.5-3.5x | Simple operations, excellent scaling |
| Histogram Equalization | 1.5-2.5x | 1.2-2.0x | Has sequential bottlenecks (CDF calculation) |

## 🎛️ Algorithm Details

### 1. Gaussian Blur (`gaussianBlur*`)
- **Purpose**: Image smoothing/noise reduction
- **Method**: Convolution with Gaussian kernel (5x5 default)
- **Parallelization**: Row-wise distribution, independent pixel processing
- **Complexity**: O(n²×k²) where n=image size, k=kernel size

### 2. Edge Detection (`edgeDetection*`)
- **Purpose**: Detect edges using Sobel operator
- **Method**: Gradient calculation in X and Y directions
- **Parallelization**: Pixel-level parallelism
- **Complexity**: O(n²)

### 3. Histogram Equalization (`histogramEqualization*`)
- **Purpose**: Enhance image contrast
- **Method**: Redistribute pixel intensities
- **Parallelization**: Parallel histogram calculation, sequential CDF
- **Complexity**: O(n²)

## 🔧 Optimization Features

### OpenMP Optimizations
- **Loop Collapse**: `collapse(2)` for nested loops
- **Scheduling**: Dynamic scheduling for load balancing
- **Reduction**: Efficient histogram accumulation
- **Thread-local Storage**: Minimize false sharing

### MPI Optimizations  
- **Domain Decomposition**: Row-wise data distribution
- **Load Balancing**: Even distribution with remainder handling
- **Communication Minimization**: Reduce data transfers
- **Overlapping**: Computation and communication overlap where possible

## 📈 Benchmarking Details

### Test Configuration
- **Image Sizes**: 512×512, 1024×1024, 2048×2048 pixels
- **Iterations**: 3 runs per test (averaged)
- **Thread/Process Count**: 4 (configurable)
- **Algorithms**: All three implementations per algorithm

### Performance Metrics
- **Execution Time**: Wall-clock time measurement
- **Speedup**: Compared to sequential baseline
- **Efficiency**: Percentage of theoretical maximum
- **Scalability**: Performance across different problem sizes

## 🐛 Troubleshooting

### Common Issues

1. **MPI not found**
   ```bash
   # Load MPI module (if using modules)
   module load mpi/openmpi
   
   # Or ensure MPI is in PATH
   export PATH=/usr/lib64/openmpi/bin:$PATH
   ```

2. **Compilation errors**
   ```bash
   # Check compiler version
   mpicxx --version
   
   # Ensure OpenMP support
   echo | mpicxx -fopenmp -dM -E - | grep -i openmp
   ```

3. **Runtime errors**
   ```bash
   # Check available cores
   nproc
   
   # Verify MPI installation
   mpirun --version
   ```

4. **Poor performance**
   - Ensure sufficient memory (2GB+ recommended)
   - Check CPU usage during execution
   - Verify no other heavy processes running
   - Try different thread/process counts

### Performance Tuning

```bash
# Experiment with different configurations
export OMP_NUM_THREADS=8
mpirun -np 2 ./parallel_image_processor

# Try different MPI binding
mpirun -np 4 --bind-to core --map-by core ./parallel_image_processor

# NUMA awareness
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

## 📝 Notes

- The system uses synthetic test images for consistent benchmarking
- All algorithms are suitable for parallel implementation as requested
- Large image processing ensures meaningful parallel workloads
- Comprehensive performance analysis provides insights for optimization

## 🔗 Further Enhancements

Potential improvements for advanced users:
- **Hybrid MPI+OpenMP**: Combine both approaches
- **GPU Acceleration**: CUDA/OpenCL implementations  
- **Advanced Algorithms**: FFT-based convolution, wavelet transforms
- **Memory Optimization**: Cache-friendly data layouts
- **I/O Integration**: Real image file support (JPEG, PNG)
