#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <mpi.h>

// Simple image structure
struct Image {
    std::vector<std::vector<int>> data;
    int width, height;
    
    Image(int w, int h) : width(w), height(h) {
        data.resize(height, std::vector<int>(width, 0));
    }
};

// Utility functions
void generateTestImage(Image& img, int seed = 42) {
    srand(seed);
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            img.data[i][j] = rand() % 256;
        }
    }
}

void saveImage(const Image& img, const std::string& filename) {
    std::ofstream file(filename);
    file << "P2\n" << img.width << " " << img.height << "\n255\n";
    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            file << img.data[i][j] << " ";
        }
        file << "\n";
    }
}

// ================================
// SEQUENTIAL ALGORITHMS
// ================================

// 1. Gaussian Blur (computationally intensive)
Image gaussianBlurSequential(const Image& input, int kernelSize = 5, double sigma = 1.0) {
    Image output(input.width, input.height);
    
    // Generate Gaussian kernel
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double sum = 0.0;
    int center = kernelSize / 2;
    
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            double x = i - center;
            double y = j - center;
            kernel[i][j] = exp(-(x*x + y*y) / (2.0 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }
    
    // Apply convolution
    for (int i = center; i < input.height - center; i++) {
        for (int j = center; j < input.width - center; j++) {
            double value = 0.0;
            for (int ki = 0; ki < kernelSize; ki++) {
                for (int kj = 0; kj < kernelSize; kj++) {
                    int pi = i + ki - center;
                    int pj = j + kj - center;
                    value += input.data[pi][pj] * kernel[ki][kj];
                }
            }
            output.data[i][j] = static_cast<int>(value);
        }
    }
    
    return output;
}

// 2. Edge Detection (Sobel operator)
Image edgeDetectionSequential(const Image& input) {
    Image output(input.width, input.height);
    
    // Sobel kernels
    int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int i = 1; i < input.height - 1; i++) {
        for (int j = 1; j < input.width - 1; j++) {
            int gx = 0, gy = 0;
            
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = input.data[i + ki][j + kj];
                    gx += pixel * sobelX[ki + 1][kj + 1];
                    gy += pixel * sobelY[ki + 1][kj + 1];
                }
            }
            
            int magnitude = static_cast<int>(sqrt(gx*gx + gy*gy));
            output.data[i][j] = std::min(255, magnitude);
        }
    }
    
    return output;
}

// 3. Histogram Equalization
Image histogramEqualizationSequential(const Image& input) {
    Image output(input.width, input.height);
    
    // Calculate histogram
    std::vector<int> histogram(256, 0);
    for (int i = 0; i < input.height; i++) {
        for (int j = 0; j < input.width; j++) {
            histogram[input.data[i][j]]++;
        }
    }
    
    // Calculate cumulative distribution
    std::vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
    
    // Normalize CDF
    int totalPixels = input.width * input.height;
    std::vector<int> equalizedValues(256);
    for (int i = 0; i < 256; i++) {
        equalizedValues[i] = static_cast<int>((cdf[i] * 255.0) / totalPixels);
    }
    
    // Apply equalization
    for (int i = 0; i < input.height; i++) {
        for (int j = 0; j < input.width; j++) {
            output.data[i][j] = equalizedValues[input.data[i][j]];
        }
    }
    
    return output;
}

// 4. Otsu Thresholding (finds optimal threshold for binarization)
Image otsuThresholdingSequential(const Image& input) {
    Image output(input.width, input.height);
    
    // Calculate histogram
    std::vector<int> histogram(256, 0);
    for (int i = 0; i < input.height; i++) {
        for (int j = 0; j < input.width; j++) {
            histogram[input.data[i][j]]++;
        }
    }
    
    int totalPixels = input.width * input.height;
    
    // Calculate total mean
    double totalMean = 0.0;
    for (int i = 0; i < 256; i++) {
        totalMean += i * histogram[i];
    }
    totalMean /= totalPixels;
    
    // Find optimal threshold using Otsu's method
    double maxVariance = 0.0;
    int optimalThreshold = 0;
    
    for (int threshold = 0; threshold < 256; threshold++) {
        // Calculate background weight and mean
        int backgroundWeight = 0;
        double backgroundMean = 0.0;
        
        for (int i = 0; i <= threshold; i++) {
            backgroundWeight += histogram[i];
            backgroundMean += i * histogram[i];
        }
        
        if (backgroundWeight == 0) continue;
        backgroundMean /= backgroundWeight;
        
        // Calculate foreground weight and mean
        int foregroundWeight = totalPixels - backgroundWeight;
        if (foregroundWeight == 0) continue;
        
        double foregroundMean = (totalMean * totalPixels - backgroundMean * backgroundWeight) / foregroundWeight;
        
        // Calculate between-class variance
        double betweenClassVariance = (double)backgroundWeight * foregroundWeight * 
                                     pow(backgroundMean - foregroundMean, 2) / (totalPixels * totalPixels);
        
        if (betweenClassVariance > maxVariance) {
            maxVariance = betweenClassVariance;
            optimalThreshold = threshold;
        }
    }
    
    // Apply thresholding
    for (int i = 0; i < input.height; i++) {
        for (int j = 0; j < input.width; j++) {
            output.data[i][j] = (input.data[i][j] > optimalThreshold) ? 255 : 0;
        }
    }
    
    return output;
}

// ================================
// OPENMP PARALLEL ALGORITHMS
// ================================

Image gaussianBlurOpenMP(const Image& input, int kernelSize = 5, double sigma = 1.0) {
    Image output(input.width, input.height);
    
    // Generate Gaussian kernel (same as sequential)
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double sum = 0.0;
    int center = kernelSize / 2;
    
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            double x = i - center;
            double y = j - center;
            kernel[i][j] = exp(-(x*x + y*y) / (2.0 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }
    
    // Parallel convolution
    #pragma omp parallel for collapse(2) schedule(dynamic, 64)
    for (int i = center; i < input.height - center; i++) {
        for (int j = center; j < input.width - center; j++) {
            double value = 0.0;
            for (int ki = 0; ki < kernelSize; ki++) {
                for (int kj = 0; kj < kernelSize; kj++) {
                    int pi = i + ki - center;
                    int pj = j + kj - center;
                    value += input.data[pi][pj] * kernel[ki][kj];
                }
            }
            output.data[i][j] = static_cast<int>(value);
        }
    }
    
    return output;
}

Image edgeDetectionOpenMP(const Image& input) {
    Image output(input.width, input.height);
    
    int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < input.height - 1; i++) {
        for (int j = 1; j < input.width - 1; j++) {
            int gx = 0, gy = 0;
            
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = input.data[i + ki][j + kj];
                    gx += pixel * sobelX[ki + 1][kj + 1];
                    gy += pixel * sobelY[ki + 1][kj + 1];
                }
            }
            
            int magnitude = static_cast<int>(sqrt(gx*gx + gy*gy));
            output.data[i][j] = std::min(255, magnitude);
        }
    }
    
    return output;
}

Image histogramEqualizationOpenMP(const Image& input) {
    Image output(input.width, input.height);
    
    // Calculate histogram in parallel
    std::vector<int> histogram(256, 0);
    
    #pragma omp parallel
    {
        std::vector<int> local_histogram(256, 0);
        
        #pragma omp for collapse(2) nowait
        for (int i = 0; i < input.height; i++) {
            for (int j = 0; j < input.width; j++) {
                local_histogram[input.data[i][j]]++;
            }
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++) {
                histogram[i] += local_histogram[i];
            }
        }
    }
    
    // Calculate CDF (sequential - small array)
    std::vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i-1] + histogram[i];
    }
    
    int totalPixels = input.width * input.height;
    std::vector<int> equalizedValues(256);
    for (int i = 0; i < 256; i++) {
        equalizedValues[i] = static_cast<int>((cdf[i] * 255.0) / totalPixels);
    }
    
    // Apply equalization in parallel
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < input.height; i++) {
        for (int j = 0; j < input.width; j++) {
            output.data[i][j] = equalizedValues[input.data[i][j]];
        }
    }
    
    return output;
}

// 4. Otsu Thresholding OpenMP
Image otsuThresholdingOpenMP(const Image& input) {
    Image output(input.width, input.height);
    
    // Calculate histogram in parallel
    std::vector<int> histogram(256, 0);
    
    #pragma omp parallel
    {
        std::vector<int> local_histogram(256, 0);
        
        #pragma omp for collapse(2) nowait
        for (int i = 0; i < input.height; i++) {
            for (int j = 0; j < input.width; j++) {
                local_histogram[input.data[i][j]]++;
            }
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++) {
                histogram[i] += local_histogram[i];
            }
        }
    }
    
    int totalPixels = input.width * input.height;
    
    // Calculate total mean
    double totalMean = 0.0;
    for (int i = 0; i < 256; i++) {
        totalMean += i * histogram[i];
    }
    totalMean /= totalPixels;
    
    // Find optimal threshold - parallelize the threshold search
    double maxVariance = 0.0;
    int optimalThreshold = 0;
    
    #pragma omp parallel
    {
        double local_maxVariance = 0.0;
        int local_optimalThreshold = 0;
        
        #pragma omp for schedule(static)
        for (int threshold = 0; threshold < 256; threshold++) {
            // Calculate background weight and mean
            int backgroundWeight = 0;
            double backgroundMean = 0.0;
            
            for (int i = 0; i <= threshold; i++) {
                backgroundWeight += histogram[i];
                backgroundMean += i * histogram[i];
            }
            
            if (backgroundWeight == 0) continue;
            backgroundMean /= backgroundWeight;
            
            // Calculate foreground weight and mean
            int foregroundWeight = totalPixels - backgroundWeight;
            if (foregroundWeight == 0) continue;
            
            double foregroundMean = (totalMean * totalPixels - backgroundMean * backgroundWeight) / foregroundWeight;
            
            // Calculate between-class variance
            double betweenClassVariance = (double)backgroundWeight * foregroundWeight * 
                                         pow(backgroundMean - foregroundMean, 2) / (totalPixels * totalPixels);
            
            if (betweenClassVariance > local_maxVariance) {
                local_maxVariance = betweenClassVariance;
                local_optimalThreshold = threshold;
            }
        }
        
        #pragma omp critical
        {
            if (local_maxVariance > maxVariance) {
                maxVariance = local_maxVariance;
                optimalThreshold = local_optimalThreshold;
            }
        }
    }
    
    // Apply thresholding in parallel
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < input.height; i++) {
        for (int j = 0; j < input.width; j++) {
            output.data[i][j] = (input.data[i][j] > optimalThreshold) ? 255 : 0;
        }
    }
    
    return output;
}

// ================================
// MPI PARALLEL ALGORITHMS
// ================================

Image gaussianBlurMPI(const Image& input, int kernelSize = 5, double sigma = 1.0) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    Image output(input.width, input.height);
    
    // Generate kernel on all processes
    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));
    double sum = 0.0;
    int center = kernelSize / 2;
    
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            double x = i - center;
            double y = j - center;
            kernel[i][j] = exp(-(x*x + y*y) / (2.0 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }
    
    // Divide work among processes (row-wise)
    int rowsPerProcess = (input.height - 2 * center) / size;
    int extraRows = (input.height - 2 * center) % size;
    
    int startRow = center + rank * rowsPerProcess + std::min(rank, extraRows);
    int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);
    
    // Each process works on its assigned rows
    for (int i = startRow; i < endRow; i++) {
        for (int j = center; j < input.width - center; j++) {
            double value = 0.0;
            for (int ki = 0; ki < kernelSize; ki++) {
                for (int kj = 0; kj < kernelSize; kj++) {
                    int pi = i + ki - center;
                    int pj = j + kj - center;
                    value += input.data[pi][pj] * kernel[ki][kj];
                }
            }
            output.data[i][j] = static_cast<int>(value);
        }
    }
    
    // Gather results
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            int pStartRow = center + p * rowsPerProcess + std::min(p, extraRows);
            int pEndRow = pStartRow + rowsPerProcess + (p < extraRows ? 1 : 0);
            
            for (int i = pStartRow; i < pEndRow; i++) {
                MPI_Recv(&output.data[i][center], input.width - 2*center, MPI_INT, 
                        p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int i = startRow; i < endRow; i++) {
            MPI_Send(&output.data[i][center], input.width - 2*center, MPI_INT, 
                    0, 0, MPI_COMM_WORLD);
        }
    }
    
    return output;
}

// MPI Otsu Thresholding
Image otsuThresholdingMPI(const Image& input) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    Image output(input.width, input.height);
    
    // Divide histogram calculation among processes
    int pixelsPerProcess = (input.height * input.width) / size;
    int extraPixels = (input.height * input.width) % size;
    
    int startPixel = rank * pixelsPerProcess + std::min(rank, extraPixels);
    int endPixel = startPixel + pixelsPerProcess + (rank < extraPixels ? 1 : 0);
    
    // Calculate local histogram
    std::vector<int> localHistogram(256, 0);
    
    for (int pixel = startPixel; pixel < endPixel; pixel++) {
        int i = pixel / input.width;
        int j = pixel % input.width;
        localHistogram[input.data[i][j]]++;
    }
    
    // Reduce histograms
    std::vector<int> globalHistogram(256, 0);
    MPI_Allreduce(localHistogram.data(), globalHistogram.data(), 256, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    int totalPixels = input.width * input.height;
    
    // Calculate total mean (all processes do this)
    double totalMean = 0.0;
    for (int i = 0; i < 256; i++) {
        totalMean += i * globalHistogram[i];
    }
    totalMean /= totalPixels;
    
    // Divide threshold search among processes
    int thresholdsPerProcess = 256 / size;
    int extraThresholds = 256 % size;
    
    int startThreshold = rank * thresholdsPerProcess + std::min(rank, extraThresholds);
    int endThreshold = startThreshold + thresholdsPerProcess + (rank < extraThresholds ? 1 : 0);
    
    // Find local optimal threshold
    double localMaxVariance = 0.0;
    int localOptimalThreshold = 0;
    
    for (int threshold = startThreshold; threshold < endThreshold; threshold++) {
        // Calculate background weight and mean
        int backgroundWeight = 0;
        double backgroundMean = 0.0;
        
        for (int i = 0; i <= threshold; i++) {
            backgroundWeight += globalHistogram[i];
            backgroundMean += i * globalHistogram[i];
        }
        
        if (backgroundWeight == 0) continue;
        backgroundMean /= backgroundWeight;
        
        // Calculate foreground weight and mean
        int foregroundWeight = totalPixels - backgroundWeight;
        if (foregroundWeight == 0) continue;
        
        double foregroundMean = (totalMean * totalPixels - backgroundMean * backgroundWeight) / foregroundWeight;
        
        // Calculate between-class variance
        double betweenClassVariance = (double)backgroundWeight * foregroundWeight * 
                                     pow(backgroundMean - foregroundMean, 2) / (totalPixels * totalPixels);
        
        if (betweenClassVariance > localMaxVariance) {
            localMaxVariance = betweenClassVariance;
            localOptimalThreshold = threshold;
        }
    }
    
    // Find global optimal threshold
    struct {
        double variance;
        int threshold;
    } localResult = {localMaxVariance, localOptimalThreshold};
    
    struct {
        double variance;
        int threshold;
    } globalResult;
    
    MPI_Allreduce(&localResult, &globalResult, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    
    int optimalThreshold = globalResult.threshold;
    
    // Apply thresholding in parallel (each process handles its pixel range)
    for (int pixel = startPixel; pixel < endPixel; pixel++) {
        int i = pixel / input.width;
        int j = pixel % input.width;
        output.data[i][j] = (input.data[i][j] > optimalThreshold) ? 255 : 0;
    }
    
    // Gather results
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            int pStartPixel = p * pixelsPerProcess + std::min(p, extraPixels);
            int pEndPixel = pStartPixel + pixelsPerProcess + (p < extraPixels ? 1 : 0);
            
            for (int pixel = pStartPixel; pixel < pEndPixel; pixel++) {
                int i = pixel / input.width;
                int j = pixel % input.width;
                MPI_Recv(&output.data[i][j], 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int pixel = startPixel; pixel < endPixel; pixel++) {
            int i = pixel / input.width;
            int j = pixel % input.width;
            MPI_Send(&output.data[i][j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    
    return output;
}

// ================================
// BENCHMARKING SYSTEM
// ================================

struct BenchmarkResult {
    double sequentialTime;
    double openmpTime;
    double mpiTime;
    double openmpSpeedup;
    double mpiSpeedup;
    std::string algorithm;
};

class ImageProcessor {
public:
    static BenchmarkResult benchmarkAlgorithm(const std::string& algorithm, 
                                             const Image& input, 
                                             int iterations = 3) {
        BenchmarkResult result;
        result.algorithm = algorithm;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        
        // Sequential timing
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            if (algorithm == "gaussian") {
                auto img = gaussianBlurSequential(input);
            } else if (algorithm == "edge") {
                auto img = edgeDetectionSequential(input);
            } else if (algorithm == "histogram") {
                auto img = histogramEqualizationSequential(input);
            } else if (algorithm == "otsu") {
                auto img = otsuThresholdingSequential(input);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        result.sequentialTime = std::chrono::duration<double>(end - start).count() / iterations;
        
        // OpenMP timing
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            if (algorithm == "gaussian") {
                auto img = gaussianBlurOpenMP(input);
            } else if (algorithm == "edge") {
                auto img = edgeDetectionOpenMP(input);
            } else if (algorithm == "histogram") {
                auto img = histogramEqualizationOpenMP(input);
            } else if (algorithm == "otsu") {
                auto img = otsuThresholdingOpenMP(input);
            }
        }
        end = std::chrono::high_resolution_clock::now();
        result.openmpTime = std::chrono::duration<double>(end - start).count() / iterations;
        
        // MPI timing (only for root process)
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            start = std::chrono::high_resolution_clock::now();
        }
        
        for (int i = 0; i < iterations; i++) {
            if (algorithm == "gaussian") {
                auto img = gaussianBlurMPI(input);
            } else if (algorithm == "otsu") {
                auto img = otsuThresholdingMPI(input);
            }
        }
        
        if (rank == 0) {
            end = std::chrono::high_resolution_clock::now();
            result.mpiTime = std::chrono::duration<double>(end - start).count() / iterations;
        }
        
        MPI_Bcast(&result.mpiTime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Calculate speedups
        result.openmpSpeedup = result.sequentialTime / result.openmpTime;
        result.mpiSpeedup = result.sequentialTime / result.mpiTime;
        
        return result;
    }
    
    static void printResults(const BenchmarkResult& result) {
        std::cout << "\n=== " << result.algorithm << " Algorithm Results ===" << std::endl;
        std::cout << "Sequential Time: " << result.sequentialTime << "s" << std::endl;
        std::cout << "OpenMP Time: " << result.openmpTime << "s" << std::endl;
        std::cout << "MPI Time: " << result.mpiTime << "s" << std::endl;
        std::cout << "OpenMP Speedup: " << result.openmpSpeedup << "x" << std::endl;
        std::cout << "MPI Speedup: " << result.mpiSpeedup << "x" << std::endl;
    }
    
    static void saveSpeedupData(const std::vector<BenchmarkResult>& results, 
                               const std::string& filename) {
        std::ofstream file(filename);
        file << "Algorithm,Sequential,OpenMP,MPI,OpenMP_Speedup,MPI_Speedup\n";
        for (const auto& result : results) {
            file << result.algorithm << ","
                 << result.sequentialTime << ","
                 << result.openmpTime << ","
                 << result.mpiTime << ","
                 << result.openmpSpeedup << ","
                 << result.mpiSpeedup << "\n";
        }
    }
};

// ================================
// MAIN FUNCTION
// ================================

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set number of OpenMP threads
    omp_set_num_threads(std::min(8, (int)omp_get_max_threads()));
    
    if (rank == 0) {
        std::cout << "Parallel Image Processing Benchmark" << std::endl;
        std::cout << "MPI Processes: " << size << std::endl;
        std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
    }
    
    // Create test images of different sizes
    std::vector<int> imageSizes = {1024, 2048, 4096, 8192};
    std::vector<BenchmarkResult> allResults;
    
    for (int imageSize : imageSizes) {
        if (rank == 0) {
            std::cout << "\n--- Testing with " << imageSize << "x" << imageSize 
                      << " image ---" << std::endl;
        }
        
        Image testImage(imageSize, imageSize);
        generateTestImage(testImage);
        
        // Benchmark each algorithm
        std::vector<std::string> algorithms = {"gaussian", "edge", "histogram", "otsu"};
        
        for (const std::string& algo : algorithms) {
            auto result = ImageProcessor::benchmarkAlgorithm(algo, testImage, 3);
            result.algorithm += "_" + std::to_string(imageSize);
            
            if (rank == 0) {
                ImageProcessor::printResults(result);
                allResults.push_back(result);
            }
        }
    }
    
    // Save results for plotting
    if (rank == 0) {
        ImageProcessor::saveSpeedupData(allResults, "speedup_results.csv");
        std::cout << "\nResults saved to speedup_results.csv" << std::endl;
        std::cout << "Use the provided Python script to generate speedup graphs." << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}