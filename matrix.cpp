#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <chrono> // For timing

using namespace std;
using namespace std::chrono;

// Function to read a matrix from a file
bool readMatrix(const string& filename, vector<vector<double>>& matrix, int n) {
    ifstream file(filename);
    if (!file.is_open()) return false;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (!(file >> matrix[i][j])) return false;
        }
    }
    return true;
}

// Function to generate a random matrix
void generateMatrix(vector<vector<double>>& matrix, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            matrix[i][j] = rand() % 10; // random numbers 0-9
}

// Function to write a matrix to a file
void writeMatrix(const string& filename, const vector<vector<double>>& matrix, int n) {
    ofstream file(filename);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            file << matrix[i][j] << " ";
        file << endl;
    }
}

// Function to perform the operation
void computeMatrix(const vector<vector<double>>& A,
                   const vector<vector<double>>& B,
                   vector<vector<double>>& C,
                   int n,
                   const string& op) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (op == "add") C[i][j] = A[i][j] + B[i][j];
            else if (op == "sub") C[i][j] = A[i][j] - B[i][j];
            else if (op == "mul") C[i][j] = A[i][j] * B[i][j];
            else if (op == "div") C[i][j] = B[i][j] != 0 ? A[i][j] / B[i][j] : 0;
            else {
                cerr << "Unknown operation: " << op << endl;
                exit(1);
            }
        }
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

    vector<vector<double>> A(n, vector<double>(n));
    vector<vector<double>> B(n, vector<double>(n));
    vector<vector<double>> C(n, vector<double>(n));

    // Try reading matrices from files, otherwise generate randomly
    if (!readMatrix("A.txt", A, n)) {
        cout << "A.txt not found. Generating random matrix A." << endl;
        generateMatrix(A, n);
    }
    if (!readMatrix("B.txt", B, n)) {
        cout << "B.txt not found. Generating random matrix B." << endl;
        generateMatrix(B, n);
    }

    // Compute C with timing
    auto start = high_resolution_clock::now();
    computeMatrix(A, B, C, n, op);
    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<duration<double>>(end - start).count();

    // Print result
    cout << "Matrix C (" << op << "):" << endl;
    cout << "Elapsed time: " << elapsed << " seconds" << endl;

    // Optionally write to file
    if (writeOutput) {
        writeMatrix("C_matrix.txt", C, n);
        cout << "Result written to C_matrix.txt" << endl;
    }

    return 0;
}
