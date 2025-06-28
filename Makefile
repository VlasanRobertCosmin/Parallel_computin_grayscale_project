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
