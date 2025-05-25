#!/bin/bash

# Define filenames
SEQ_SRC="grayscale_seq.cpp"
OMP_SRC="grayscale_omp.cpp"
MPI_SRC="grayscale_mpi.cpp"

SEQ_EXE="grayscale_seq"
OMP_EXE="grayscale_omp"
MPI_EXE="grayscale_mpi"

# Clean up old executables
rm -f $SEQ_EXE $OMP_EXE $MPI_EXE timing_results.csv

# Compile sequential
echo "Compiling sequential..."
g++ -O2 $SEQ_SRC -o $SEQ_EXE `pkg-config --cflags --libs opencv4`

# Compile OpenMP
echo "Compiling OpenMP..."
g++ -O2 -fopenmp $OMP_SRC -o $OMP_EXE `pkg-config --cflags --libs opencv4`

# Compile MPI
echo "Compiling MPI..."
mpic++ -O2 $MPI_SRC -o $MPI_EXE `pkg-config --cflags --libs opencv4`

# Run sequential once
echo "Running sequential..."
./$SEQ_EXE

# Detect max cores
MAX_CORES=$(nproc)
echo "Detected $MAX_CORES cores"

# Run OpenMP tests
echo "Running OpenMP tests..."
for threads in 1 2 4 8 16
do
    if [ $threads -le $MAX_CORES ]; then
        export OMP_NUM_THREADS=$threads
        echo "Running OpenMP with $threads threads..."
        ./$OMP_EXE
    fi
done

MAX_CORES=$(nproc)

for procs in 1 2 4 8 16
do
    if [ $procs -le $MAX_CORES ]; then
        mpirun -np $procs ./$MPI_EXE
    else
        echo "Skipping $procs processes (only $MAX_CORES cores available)"
    fi
done
echo "All runs completed. Results saved in timing_results.csv"
