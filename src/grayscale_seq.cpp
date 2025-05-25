#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat image = cv::imread("data/2.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not open image.\n";
        return 1;
    }

    cv::Mat gray(image.rows, image.cols, CV_8UC1);
    long long sumR = 0, sumG = 0, sumB = 0;
    int histogramR[256] = {0}, histogramG[256] = {0}, histogramB[256] = {0};

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            uchar B = pixel[0], G = pixel[1], R = pixel[2];

            gray.at<uchar>(i, j) = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
            histogramR[R]++; histogramG[G]++; histogramB[B]++;
            sumR += R; sumG += G; sumB += B;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double numPixels = image.rows * image.cols;

    std::cout << "Average R: " << sumR / numPixels << "\n";
    std::cout << "Average G: " << sumG / numPixels << "\n";
    std::cout << "Average B: " << sumB / numPixels << "\n";
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    cv::imwrite("output/grayscale_seq.jpg", gray);

    std::ofstream file("timing_results.csv", std::ios::app);
    file << "seq,1," << elapsed.count() << "\n";
    file.close();

    return 0;
}


//run instructions
//make grayscale_seq
//./grayscale_seq