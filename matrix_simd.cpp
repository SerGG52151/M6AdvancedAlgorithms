#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <immintrin.h> // For AVX intrinsics
#include <cstring>     // For memset
#include <chrono>      // For timing

using namespace std;
using namespace std::chrono;

// Read a matrix from file
bool readMatrix(const string& filename, vector<float>& matrix, int n) {
    ifstream file(filename);
    if (!file.is_open()) return false;

    for (int i = 0; i < n * n; ++i) {
        if (!(file >> matrix[i])) return false;
    }
    return true;
}

// Generate random matrix
void generateMatrix(vector<float>& matrix, int n) {
    for (int i = 0; i < n * n; ++i)
        matrix[i] = static_cast<float>(rand() % 10);
}

// Write matrix to file
void writeMatrix(const string& filename, const vector<float>& matrix, int n) {
    ofstream file(filename);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            file << matrix[i * n + j] << " ";
        file << endl;
    }
}

// SIMD compute
void computeSIMD(const vector<float>& A, const vector<float>& B, vector<float>& C, int n, const string& op) {
    int simdWidth = 8; // AVX: 8 floats per __m256
    int size = n * n;

    int i = 0;
    for (; i <= size - simdWidth; i += simdWidth) {
        __m256 aVec = _mm256_loadu_ps(&A[i]);
        __m256 bVec = _mm256_loadu_ps(&B[i]);
        __m256 cVec;

        if (op == "add") cVec = _mm256_add_ps(aVec, bVec);
        else if (op == "sub") cVec = _mm256_sub_ps(aVec, bVec);
        else if (op == "mul") cVec = _mm256_mul_ps(aVec, bVec);
        else if (op == "div") {
            __m256 mask = _mm256_cmp_ps(bVec, _mm256_set1_ps(0.0f), _CMP_EQ_OQ);
            __m256 safeB = _mm256_blendv_ps(bVec, _mm256_set1_ps(1.0f), mask); // replace 0 with 1
            cVec = _mm256_div_ps(aVec, safeB);
        } else {
            cerr << "Unknown operation: " << op << endl;
            exit(1);
        }

        _mm256_storeu_ps(&C[i], cVec);
    }

    // Handle tail elements
    for (; i < size; ++i) {
        if (op == "add") C[i] = A[i] + B[i];
        else if (op == "sub") C[i] = A[i] - B[i];
        else if (op == "mul") C[i] = A[i] * B[i];
        else if (op == "div") C[i] = B[i] != 0 ? A[i] / B[i] : 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " n operation [write_output]" << endl;
        cerr << "operation: add | sub | mul | div" << endl;
        cerr << "write_output: optional, 1 to write C.txt" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    string op = argv[2];
    bool writeOutput = (argc > 3) && (atoi(argv[3]) == 1);

    srand(time(0));

    vector<float> A(n * n);
    vector<float> B(n * n);
    vector<float> C(n * n, 0.0f);

    // Read or generate
    if (!readMatrix("A.txt", A, n)) {
        cout << "A.txt not found. Generating random matrix A." << endl;
        generateMatrix(A, n);
    }
    if (!readMatrix("B.txt", B, n)) {
        cout << "B.txt not found. Generating random matrix B." << endl;
        generateMatrix(B, n);
    }

    // Compute with timing
    auto start = high_resolution_clock::now();
    computeSIMD(A, B, C, n, op);
    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<duration<double>>(end - start).count();

    // Print result
    cout << "Matrix C (" << op << "):" << endl;
    cout << "Elapsed time: " << elapsed << " seconds" << endl;

    if (writeOutput) {
        writeMatrix("C_matrix_simd.txt", C, n);
        cout << "Result written to C_matrix_simd.txt" << endl;
    }

    return 0;
}
