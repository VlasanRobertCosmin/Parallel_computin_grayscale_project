#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double total_start = MPI_Wtime();

    cv::Mat image;
    int rows = 0, cols = 0;
    if (rank == 0) {
        image = cv::imread("data/2.png", cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Could not open image.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        rows = image.rows;
        cols = image.cols;
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = rows / size;
    int remaining_rows = rows % size;
    int start_row = rank * local_rows + std::min(rank, remaining_rows);
    int actual_rows = local_rows + (rank < remaining_rows ? 1 : 0);
    int local_size = actual_rows * cols;

    cv::Mat local_color(actual_rows, cols, CV_8UC3);
    cv::Mat local_gray(actual_rows, cols, CV_8UC1);

    if (rank == 0) {
        for (int p = 1; p < size; ++p) {
            int s_row = p * local_rows + std::min(p, remaining_rows);
            int a_rows = local_rows + (p < remaining_rows ? 1 : 0);
            MPI_Send(image.ptr(s_row), a_rows * cols * 3, MPI_UNSIGNED_CHAR, p, 0, MPI_COMM_WORLD);
        }
        local_color = image.rowRange(start_row, start_row + actual_rows).clone();
    } else {
        MPI_Recv(local_color.ptr(), local_size * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    long long local_sumR = 0, local_sumG = 0, local_sumB = 0;
    int histR[256] = {0}, histG[256] = {0}, histB[256] = {0};

    for (int i = 0; i < actual_rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv::Vec3b pixel = local_color.at<cv::Vec3b>(i, j);
            uchar B = pixel[0], G = pixel[1], R = pixel[2];
            local_gray.at<uchar>(i, j) = static_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
            histR[R]++; histG[G]++; histB[B]++;
            local_sumR += R; local_sumG += G; local_sumB += B;
        }
    }

    long long totalR = 0, totalG = 0, totalB = 0;
    int totalHistR[256] = {0}, totalHistG[256] = {0}, totalHistB[256] = {0};

    MPI_Reduce(&local_sumR, &totalR, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sumG, &totalG, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sumB, &totalB, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&histR, &totalHistR, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&histG, &totalHistG, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&histB, &totalHistB, 256, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cv::Mat final_gray(rows, cols, CV_8UC1);
        local_gray.copyTo(final_gray.rowRange(start_row, start_row + actual_rows));
        for (int p = 1; p < size; ++p) {
            int s_row = p * local_rows + std::min(p, remaining_rows);
            int a_rows = local_rows + (p < remaining_rows ? 1 : 0);
            cv::Mat recv_part(a_rows, cols, CV_8UC1);
            MPI_Recv(recv_part.ptr(), a_rows * cols, MPI_UNSIGNED_CHAR, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_part.copyTo(final_gray.rowRange(s_row, s_row + a_rows));
        }

        double total_pixels = rows * cols;
        std::cout << "Average R: " << totalR / total_pixels << "\n";
        std::cout << "Average G: " << totalG / total_pixels << "\n";
        std::cout << "Average B: " << totalB / total_pixels << "\n";

        cv::imwrite("output/grayscale_mpi.jpg", final_gray);

        double total_end = MPI_Wtime();
        std::cout << "Execution time (MPI): " << (total_end - total_start) << " seconds\n";

        std::ofstream file("timing_results.csv", std::ios::app);
        file << "mpi," << size << "," << (total_end - total_start) << "\n";
        file.close();
    } else {
        MPI_Send(local_gray.ptr(), local_size, MPI_UNSIGNED_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}


/*
make grayscale_mpi

mpirun -np 4 ./grayscale_mpi
*/