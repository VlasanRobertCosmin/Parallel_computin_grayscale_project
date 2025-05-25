CXX = g++
CXXFLAGS = -O2 `pkg-config --cflags opencv4`
LDLIBS = `pkg-config --libs opencv4`
MPICXX = mpic++

all: grayscale_seq grayscale_omp grayscale_mpi

grayscale_seq: src/grayscale_seq.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

grayscale_omp: src/grayscale_omp.cpp
	$(CXX) -fopenmp $(CXXFLAGS) -o $@ $^ $(LDLIBS)

grayscale_mpi: src/grayscale_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)


clean:
	rm -f grayscale_seq grayscale_omp grayscale_mpi
