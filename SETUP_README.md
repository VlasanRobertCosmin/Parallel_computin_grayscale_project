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
