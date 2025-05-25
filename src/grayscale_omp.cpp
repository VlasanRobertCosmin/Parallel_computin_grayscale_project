#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <omp.h>

int main() {
    cv::Mat image = cv::imread("data/2.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not open image.\n";
        return 1;
    }

    double start_time = omp_get_wtime();

    cv::Mat gray(image.rows, image.cols, CV_8UC1);
    int rows = image.rows, cols = image.cols;
    long long sumR = 0, sumG = 0, sumB = 0;
    int histogramR[256] = {0}, histogramG[256] = {0}, histogramB[256] = {0};
    const int num_threads = omp_get_max_threads();
    int histR_local[num_threads][256] = {{0}}, histG_local[num_threads][256] = {{0}}, histB_local[num_threads][256] = {{0}};

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long localR = 0, localG = 0, localB = 0;
        #pragma omp for collapse(2)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
                uchar B = pixel[0], G = pixel[1], R = pixel[2];
                gray.at<uchar>(i, j) = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
                localR += R; localG += G; localB += B;
                histR_local[tid][R]++; histG_local[tid][G]++; histB_local[tid][B]++;
            }
        }
        #pragma omp atomic
        sumR += localR;
        #pragma omp atomic
        sumG += localG;
        #pragma omp atomic
        sumB += localB;
    }

    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < 256; ++i) {
            histogramR[i] += histR_local[t][i];
            histogramG[i] += histG_local[t][i];
            histogramB[i] += histB_local[t][i];
        }
    }

    double end_time = omp_get_wtime();
    double numPixels = rows * cols;

    std::cout << "Average R: " << sumR / numPixels << "\n";
    std::cout << "Average G: " << sumG / numPixels << "\n";
    std::cout << "Average B: " << sumB / numPixels << "\n";
    std::cout << "Execution time (OpenMP): " << (end_time - start_time) << " seconds\n";

    cv::imwrite("output/grayscale_omp.jpg", gray);

    std::ofstream file("timing_results.csv", std::ios::app);
    file << "omp," << num_threads << "," << (end_time - start_time) << "\n";
    file.close();

    return 0;
}
