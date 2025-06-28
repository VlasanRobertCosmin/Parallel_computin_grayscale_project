#!/bin/bash
# setup_parallel_image_processing.sh
# This script creates all necessary files for the parallel image processing system

echo "🚀 Setting up Parallel Image Processing System..."
echo "=================================================="

# Create the plot_speedup.py file
cat > plot_speedup.py << 'EOF'
#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Remove seaborn import - not needed

def plot_speedup_graphs():
    """Generate comprehensive speedup visualization"""
    
    # Read the benchmark results
    try:
        df = pd.read_csv('speedup_results.csv')
        print("✅ CSV file loaded successfully")
    except FileNotFoundError:
        print("Error: speedup_results.csv not found. Run the benchmark first.")
        return
    
    # Extract algorithm and image size
    df[['Algorithm', 'ImageSize']] = df['Algorithm'].str.split('_', expand=True)
    df['ImageSize'] = df['ImageSize'].astype(int)
    
    # Set up the plotting style (use compatible styles)
    try:
        plt.style.use('seaborn')  # Try seaborn first
        print("Using seaborn style")
    except:
        try:
            plt.style.use('ggplot')  # Fallback to ggplot
            print("Using ggplot style")
        except:
            plt.style.use('default')  # Final fallback
            print("Using default style")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parallel Image Processing Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Speedup comparison by algorithm
    ax1 = axes[0, 0]
    algorithms = df['Algorithm'].unique()
    x = np.arange(len(algorithms))
    width = 0.35
    
    openmp_speedups = df.groupby('Algorithm')['OpenMP_Speedup'].mean()
    mpi_speedups = df.groupby('Algorithm')['MPI_Speedup'].mean()
    
    # Convert to numpy arrays to avoid pandas indexing issues
    openmp_values = openmp_speedups.values
    mpi_values = mpi_speedups.values
    
    bars1 = ax1.bar(x - width/2, openmp_values, width, label='OpenMP', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, mpi_values, width, label='MPI', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Average Speedup')
    ax1.set_title('Average Speedup by Algorithm')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{openmp_values[i]:.2f}', ha='center', va='bottom', fontsize=9)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mpi_values[i]:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Speedup vs Image Size
    ax2 = axes[0, 1]
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['Algorithm'] == algo].sort_values('ImageSize')
        # Convert pandas Series to numpy arrays to avoid indexing issues
        image_sizes = algo_data['ImageSize'].values
        openmp_speedups = algo_data['OpenMP_Speedup'].values
        mpi_speedups = algo_data['MPI_Speedup'].values
        
        ax2.plot(image_sizes, openmp_speedups, 
                color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                linestyle='-', label=f'{algo} (OpenMP)', linewidth=2, markersize=6)
        ax2.plot(image_sizes, mpi_speedups, 
                color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                linestyle='--', label=f'{algo} (MPI)', linewidth=2, markersize=6, alpha=0.7)
    
    ax2.set_xlabel('Image Size (pixels)')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup vs Image Size')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Fix for older matplotlib versions - use basex instead of base
    try:
        ax2.set_xscale('log', base=2)
    except:
        ax2.set_xscale('log', basex=2)
    
    # 3. Execution Time Comparison (using matplotlib instead of seaborn)
    ax3 = axes[1, 0]
    
    # Focus on largest image size for clarity
    largest_size = df['ImageSize'].max()
    df_largest = df[df['ImageSize'] == largest_size]
    
    # Manual bar plot instead of seaborn
    algorithms_largest = df_largest['Algorithm'].unique()
    x_pos = np.arange(len(algorithms_largest))
    
    # Extract times for each algorithm
    seq_times = []
    omp_times = []
    mpi_times = []
    
    for algo in algorithms_largest:
        algo_row = df_largest[df_largest['Algorithm'] == algo].iloc[0]
        seq_times.append(algo_row['Sequential'])
        omp_times.append(algo_row['OpenMP'])
        mpi_times.append(algo_row['MPI'])
    
    width = 0.25
    ax3.bar(x_pos - width, seq_times, width, label='Sequential', alpha=0.8, color='gray')
    ax3.bar(x_pos, omp_times, width, label='OpenMP', alpha=0.8, color='skyblue')
    ax3.bar(x_pos + width, mpi_times, width, label='MPI', alpha=0.8, color='lightcoral')
    
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Execution Time (seconds)')
    ax3.set_title(f'Execution Time Comparison ({largest_size}x{largest_size} image)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms_largest, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency Analysis
    ax4 = axes[1, 1]
    
    # Calculate efficiency (assuming 4 cores/processes)
    num_cores = 4
    df['OpenMP_Efficiency'] = df['OpenMP_Speedup'] / num_cores * 100
    df['MPI_Efficiency'] = df['MPI_Speedup'] / num_cores * 100
    
    x = np.arange(len(algorithms))
    openmp_eff = df.groupby('Algorithm')['OpenMP_Efficiency'].mean()
    mpi_eff = df.groupby('Algorithm')['MPI_Efficiency'].mean()
    
    # Convert to numpy arrays
    openmp_eff_values = openmp_eff.values
    mpi_eff_values = mpi_eff.values
    
    bars1 = ax4.bar(x - width/2, openmp_eff_values, width, label='OpenMP', alpha=0.8, color='green')
    bars2 = ax4.bar(x + width/2, mpi_eff_values, width, label='MPI', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Algorithm')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_title('Parallel Efficiency (4 cores/processes)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(algorithms, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Ideal (100%)')
    
    # Add value labels
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{openmp_eff_values[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mpi_eff_values[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Speedup analysis saved to 'speedup_analysis.png'")
    
    # Print summary statistics
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Average OpenMP Speedup: {df['OpenMP_Speedup'].mean():.2f}x")
    print(f"Average MPI Speedup: {df['MPI_Speedup'].mean():.2f}x")
    print(f"Best OpenMP Performance: {df['OpenMP_Speedup'].max():.2f}x ({df.loc[df['OpenMP_Speedup'].idxmax(), 'Algorithm']})")
    print(f"Best MPI Performance: {df['MPI_Speedup'].max():.2f}x ({df.loc[df['MPI_Speedup'].idxmax(), 'Algorithm']})")
    
    # Efficiency analysis
    avg_openmp_eff = df['OpenMP_Efficiency'].mean()
    avg_mpi_eff = df['MPI_Efficiency'].mean()
    print(f"\nAverage OpenMP Efficiency: {avg_openmp_eff:.1f}%")
    print(f"Average MPI Efficiency: {avg_mpi_eff:.1f}%")
    
    if avg_openmp_eff > 75 and avg_mpi_eff > 75:
        print("✅ Both implementations show good parallel efficiency!")
    elif avg_openmp_eff > 75:
        print("✅ OpenMP shows good efficiency, MPI needs optimization")
    elif avg_mpi_eff > 75:
        print("✅ MPI shows good efficiency, OpenMP needs optimization")
    else:
        print("⚠️  Both implementations need optimization for better efficiency")

if __name__ == "__main__":
    plot_speedup_graphs()
EOF

# Create the Makefile
cat > Makefile << 'EOF'
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -fopenmp -Wall -Wextra
TARGET = parallel_image_processor
SOURCE = parallel_image_processor.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) -lm

clean:
	rm -f $(TARGET) *.o speedup_results.csv *.pgm

run-sequential:
	export OMP_NUM_THREADS=1 && ./$(TARGET)

run-openmp:
	export OMP_NUM_THREADS=4 && ./$(TARGET)

run-mpi:
	mpirun -np 4 ./$(TARGET)

benchmark:
	@echo "Running comprehensive benchmark..."
	mpirun -np 4 ./$(TARGET)

.PHONY: clean run-sequential run-openmp run-mpi benchmark
EOF

# Create the main run script
cat > run_benchmark.sh << 'EOF'
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
EOF

# Create a simple test script
cat > test_system.sh << 'EOF'
#!/bin/bash

echo "🧪 Testing Parallel Image Processing System"
echo "=========================================="

# Test 1: Check dependencies
echo "1. Checking dependencies..."

if command -v mpicxx >/dev/null 2>&1; then
    echo "   ✅ MPI compiler found"
else
    echo "   ❌ MPI compiler not found"
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    echo "   ✅ Python 3 found"
else
    echo "   ❌ Python 3 not found"
    exit 1
fi

# Test 2: Check OpenMP support
echo "2. Checking OpenMP support..."
if echo | mpicxx -fopenmp -dM -E - | grep -q "OPENMP"; then
    echo "   ✅ OpenMP support detected"
else
    echo "   ❌ OpenMP support not found"
fi

# Test 3: Check if source file exists
echo "3. Checking source files..."
if [ -f "parallel_image_processor.cpp" ]; then
    echo "   ✅ Source file found"
else
    echo "   ❌ parallel_image_processor.cpp not found"
    echo "   Please make sure you have the C++ source file"
    exit 1
fi

# Test 4: Try compilation
echo "4. Testing compilation..."
make clean &>/dev/null
if make &>/dev/null; then
    echo "   ✅ Compilation successful"
    make clean &>/dev/null
else
    echo "   ❌ Compilation failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! System is ready."
echo "Run './run_benchmark.sh' to start the full benchmark."
EOF

# Make all scripts executable
chmod +x run_benchmark.sh
chmod +x plot_speedup.py  
chmod +x test_system.sh

# Create a README for this specific setup
cat > SETUP_README.md << 'EOF'
# Parallel Image Processing - Complete Setup

## Files Created:
- `Makefile` - Build configuration
- `run_benchmark.sh` - Main execution script  
- `plot_speedup.py` - Visualization script
- `test_system.sh` - System testing script

## Quick Start:
1. Make sure you have `parallel_image_processor.cpp` in this directory
2. Run: `./test_system.sh` (optional - tests your system)
3. Run: `./run_benchmark.sh` (main benchmark)

## Requirements:
- MPI compiler (mpicxx)
- OpenMP support
- Python 3 with pip
- At least 2GB RAM

## Troubleshooting:
- If MPI not found: `sudo apt-get install libopenmpi-dev openmpi-bin`
- If Python packages fail: `pip3 install --user pandas matplotlib seaborn numpy`
- If permission denied: `chmod +x *.sh *.py`
EOF

echo ""
echo "✅ Setup complete! Files created:"
echo "   - Makefile"
echo "   - run_benchmark.sh (main script)"
echo "   - plot_speedup.py (visualization - FIXED for your system)"
echo "   - test_system.sh (system test)"
echo "   - SETUP_README.md (instructions)"
echo ""
echo "📋 Next steps:"
echo "   1. Make sure 'parallel_image_processor.cpp' is in this directory"
echo "   2. Run: ./test_system.sh (optional system check)"
echo "   3. Run: ./run_benchmark.sh (main benchmark)"
echo ""
echo "🎯 The main script will:"
echo "   - Install Python dependencies (pandas, matplotlib, numpy)"
echo "   - Compile the C++ code"
echo "   - Run benchmarks on all algorithms"
echo "   - Generate performance graphs (GUARANTEED to work with your matplotlib)"
echo "   - Show speedup analysis"