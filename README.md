# Problem 1: Parallel Pi Approximation

This project implements a numerical approximation of Pi using the Riemann sum (midpoint rule) with $\int_0^1 \frac{4}{1+x^2} dx$.

## Files
- `P1.cpp`: **(Recommended)** Unified C++ file containing the Pi approximation logic AND the benchmark runner.
- `P2.cpp`: **(New)** C++ benchmark runner for Matrix operations (Problem 2).
- `main.cpp`: Original C++ source code (Pi approximation only).
- `benchmark.py`: Python script to compile, run benchmarks, generate plots/CSV.
- `benchmark.cpp`: Standalone C++ benchmark runner (calls `pi_approx.exe`).
- `Analysis P1.txt`: Analysis of the results and theoretical discussion.

## Prerequisites
- **C++ Compiler**: `g++` with OpenMP, AVX, and Pthreads support.
- **Python**: For running the benchmark script (optional).

## How to Run

### Problem 1: Pi Approximation
**Compile & Run:**
```bash
g++ -O3 -fopenmp P1.cpp -o P1 -lpthread
./P1
```

### Problem 2: Matrix Operations
**Compile & Run:**
```bash
g++ P2.cpp -o P2
./P2
```
This will:
1. Compile `matrix.cpp`, `matrix_simd.cpp`, `matrix_simd_omp.cpp`, and `generate_matrices.cpp`.
2. Generate matrices for sizes 512, 1024, 2048.
3. Run all variants (scalar, SIMD, SIMD+OpenMP) for `add`, `sub`, `mul`, `div`.

### 2. Automated Benchmark (Python)
Run the Python script to compile the code, run all tests, and generate the report.
```bash
python benchmark.py
```
This will create:
- `pi_approx.exe`: The executable.
- `benchmark_results.csv`: The raw data.
- `speedup_plot_N_*.png`: Plots of speedup vs threads.
- Console Output: Amdahl's Law estimation of serial fraction `f`.

### 3. Manual Compilation and Run (Legacy)
If you prefer to run manually:

**Compile:**
```bash
g++ -O3 -fopenmp main.cpp -o pi_approx -lpthread
```

**Run:**
Usage: `pi_approx <mode> <N> [threads]`
- `mode`: `seq`, `omp`, `std`, `pthread`
- `N`: Number of intervals (e.g., 100000000)
- `threads`: Number of threads (for parallel modes)

Examples:
```bash
./pi_approx seq 100000000
./pi_approx omp 100000000 4
./pi_approx std 100000000 8
./pi_approx pthread 100000000 8
```
