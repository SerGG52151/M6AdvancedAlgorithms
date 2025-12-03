#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Generate a random matrix of size n x n
vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> mat(n, vector<int>(n));
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            mat[i][j] = rand() % 10; // random number 0-9
    return mat;
}

// Write a matrix to a file
void writeMatrix(const string &filename, const vector<vector<int>> &mat) {
    ofstream file(filename);
    for(const auto &row : mat) {
        for(const auto &val : row)
            file << val << " ";
        file << endl;
    }
}

int main() {
    srand(time(0));

    int n;
    cout << "Enter matrix size n: ";
    cin >> n;

    vector<vector<int>> A = generateMatrix(n);
    vector<vector<int>> B = generateMatrix(n);

    writeMatrix("A.txt", A);
    writeMatrix("B.txt", B);

    cout << "Matrices A and B of size " << n << "x" << n << " generated as A.txt and B.txt" << endl;

    return 0;
}
