#include <iostream>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <K (in millions)>\n";
        return 1;
    }

    // Parse command line argument K
    int K = std::atoi(argv[1]);
    if (K <= 0) {
        std::cerr << "K must be a positive integer.\n";
        return 1;
    }

    // Calculate the number of elements (K million)
    long long N = static_cast<long long>(K) * 1000000;

    // Allocate memory for the arrays
    // TODO: adjust data type (e.g., float, double, int) if needed
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Memory allocation failed!\n";
        // Make sure to free any partially allocated memory before exiting
        free(A); free(B); free(C);
        return 1;
    }

    // Initialize arrays A and B with dummy data
    for (long long i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }


    // ---- Warm-up ----
    for (long long i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }


    // ---- Start Profiling ----
    auto start = std::chrono::high_resolution_clock::now();

    for (long long i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }


    // ---- Stop Profiling ----
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print the time taken
    std::cout << "Execution time for K=" << K << " million: " << duration.count() << " seconds\n";

    // Free the memory at the end of the program
    free(A);
    free(B);
    free(C);

    return 0;
}
