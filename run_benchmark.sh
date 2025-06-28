#!/bin/bash

echo "==================================="
echo "Parallel Image Processing Benchmark"
echo "==================================="

# Check if required tools are available
command -v mpicxx >/dev/null 2>&1 || { echo "Error: mpicxx not found. Install OpenMPI." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 not found." >&2; exit 1; }

# Install required Python packages
echo "Installing required Python packages..."
pip3 install pandas matplotlib numpy --user --quiet

# Check if source file exists
if [ ! -f "parallel_image_processor.cpp" ]; then
    echo "Error: parallel_image_processor.cpp not found!"
    echo "Please make sure you have the C++ source file in this directory."
    exit 1
fi

# Build the program
echo "Building the parallel image processor..."
make clean
make

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build successful!"

# Run the benchmark
echo "Running comprehensive benchmark..."
echo "This may take several minutes depending on your system..."

# Set optimal thread count for OpenMP
export OMP_NUM_THREADS=4

# Run with MPI
mpirun -np 4 ./parallel_image_processor

if [ $? -eq 0 ]; then
    echo "Benchmark completed successfully!"
    
    # Generate plots
    echo "Generating speedup graphs..."
    python3 plot_speedup.py
    
    echo "✅ Results saved to speedup_analysis.png"
    echo "✅ Raw data available in speedup_results.csv"
    echo ""
    echo "Open speedup_analysis.png to view the performance graphs!"
else
    echo "Benchmark failed!"
    exit 1
fi
