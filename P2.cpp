#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>

using namespace std;

// Helper to execute commands safely
int run_command(const string& cmd) {
    cout << "[Exec] " << cmd << endl;
    int ret = system(cmd.c_str());
    if (ret != 0) {
        cerr << "Command failed: " << cmd << endl;
    }
    return ret;
}

int main() {
    cout << "=========================================" << endl;
    cout << "      P2 Benchmark Runner (Matrix)       " << endl;
    cout << "=========================================" << endl;

    // 1. Compilation
    cout << "\n[Phase 1] Compiling executables..." << endl;
    
    #ifdef _WIN32
        string exe_ext = ".exe";
        string path_prefix = ""; // Windows cmd typically finds exe in current dir
    #else
        string exe_ext = "";
        string path_prefix = "./";
    #endif

    if (run_command("g++ -o matrix" + exe_ext + " matrix.cpp") != 0) return 1;
    if (run_command("g++ -O2 -mavx -o matrix_simd" + exe_ext + " matrix_simd.cpp") != 0) return 1;
    if (run_command("g++ -O2 -mavx -fopenmp -o matrix_simd_omp" + exe_ext + " matrix_simd_omp.cpp") != 0) return 1;
    if (run_command("g++ -o generate_matrices" + exe_ext + " generate_matrices.cpp") != 0) return 1;

    cout << "Compilation successful." << endl;

    // 2. Benchmarking
    cout << "\n[Phase 2] Running Benchmarks..." << endl;

    vector<int> sizes = {512, 1024, 2048};
    vector<string> ops = {"add", "sub", "mul", "div"};
    vector<string> variants = {"matrix", "matrix_simd", "matrix_simd_omp"};

    for (int N : sizes) {
        cout << "\n-----------------------------------------" << endl;
        cout << " Processing Matrix Size: " << N << endl;
        cout << "-----------------------------------------" << endl;

        // Generate matrices for size N
        // Note: generate_matrices reads N from stdin
        string gen_cmd;
        #ifdef _WIN32
            // Windows cmd: echo N | generate_matrices
            gen_cmd = "echo " + to_string(N) + " | " + path_prefix + "generate_matrices" + exe_ext;
        #else
            // Unix: echo N | ./generate_matrices
            gen_cmd = "echo " + to_string(N) + " | " + path_prefix + "generate_matrices" + exe_ext;
        #endif
        
        if (run_command(gen_cmd) != 0) {
            cerr << "Failed to generate matrices for size " << N << endl;
            continue;
        }

        // Run operations
        for (const string& op : ops) {
            cout << "\n  Operation: " << op << endl;
            for (const string& var : variants) {
                // Command: ./matrix N op 1
                string cmd = path_prefix + var + exe_ext + " " + to_string(N) + " " + op + " 1";
                run_command(cmd);
            }
        }
    }

    cout << "\nAll tests completed." << endl;
    return 0;
}
