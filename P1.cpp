#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <string>
#include <thread>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <fstream>
#include <map>
#include <algorithm>
#include <cstring>

// Include OpenMP header if available
#ifdef _OPENMP
#include <omp.h>
#endif

// ==========================================
// USER CONFIGURATION (Default values for single run)
const std::string DEFAULT_MODE = "seq";  // Options: "seq", "omp", "std", "pthread"
const long long DEFAULT_N = 10000000;    // Number of intervals
const int DEFAULT_THREADS = 4;           // Number of threads (for parallel modes)
// ==========================================

// Constants
const double PI_REF = 3.14159265358979323846;

// Function to calculate f(x) = 4 / (1 + x^2)
inline double f(double x) {
    return 4.0 / (1.0 + x * x);
}

// ---------------------------------------------------------
// PART 1: Pi Approximation Implementations
// ---------------------------------------------------------

// 1.a) Sequential baseline
double sequential_pi(long long N) {
    double h = 1.0 / N;
    double sum = 0.0;
    for (long long i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        sum += f(x);
    }
    return sum * h;
}

// 1.b) OpenMP implementation
double omp_pi(long long N, int threads) {
    double h = 1.0 / N;
    double sum = 0.0;

#ifdef _OPENMP
    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; ++i) {
        double x = (i + 0.5) * h;
        sum += f(x);
    }
#else
    std::cerr << "Warning: OpenMP not enabled. Running sequentially." << std::endl;
    return sequential_pi(N);
#endif

    return sum * h;
}

// 1.b) POSIX Threads implementation
struct PthreadData {
    long long start;
    long long end;
    double h;
    double partial_sum;
};

void* pthread_worker(void* arg) {
    PthreadData* data = (PthreadData*)arg;
    double local_sum = 0.0;
    for (long long i = data->start; i < data->end; ++i) {
        double x = (i + 0.5) * data->h;
        local_sum += f(x);
    }
    data->partial_sum = local_sum;
    return nullptr;
}

double pthread_pi(long long N, int num_threads) {
    double h = 1.0 / N;
    std::vector<pthread_t> threads(num_threads);
    std::vector<PthreadData> thread_data(num_threads);

    long long chunk_size = N / num_threads;
    long long remainder = N % num_threads;

    long long current_start = 0;
    for (int i = 0; i < num_threads; ++i) {
        long long current_end = current_start + chunk_size + (i < remainder ? 1 : 0);
        
        thread_data[i].start = current_start;
        thread_data[i].end = current_end;
        thread_data[i].h = h;
        thread_data[i].partial_sum = 0.0;

        int rc = pthread_create(&threads[i], nullptr, pthread_worker, (void*)&thread_data[i]);
        if (rc) {
            std::cerr << "Error: unable to create thread," << rc << std::endl;
            exit(-1);
        }
        current_start = current_end;
    }

    double total_sum = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
        total_sum += thread_data[i].partial_sum;
    }

    return total_sum * h;
}

// 1.b) C++ std::thread implementation
void thread_worker(long long start, long long end, double h, double& partial_sum) {
    double local_sum = 0.0;
    for (long long i = start; i < end; ++i) {
        double x = (i + 0.5) * h;
        local_sum += f(x);
    }
    partial_sum = local_sum;
}

double std_thread_pi(long long N, int num_threads) {
    double h = 1.0 / N;
    std::vector<std::thread> threads;
    std::vector<double> partial_sums(num_threads, 0.0);

    long long chunk_size = N / num_threads;
    long long remainder = N % num_threads;

    long long current_start = 0;
    for (int i = 0; i < num_threads; ++i) {
        long long current_end = current_start + chunk_size + (i < remainder ? 1 : 0);
        threads.emplace_back(thread_worker, current_start, current_end, h, std::ref(partial_sums[i]));
        current_start = current_end;
    }

    for (auto& t : threads) {
        t.join();
    }

    double total_sum = 0.0;
    for (double s : partial_sums) {
        total_sum += s;
    }

    return total_sum * h;
}

// ---------------------------------------------------------
// PART 2: Benchmarking Logic
// ---------------------------------------------------------

struct RunResult {
    std::string mode;
    long long n;
    int threads;
    double time;
    double pi_approx;
    double error;
    double speedup;
};

// Helper to run a single experiment and return results
RunResult run_experiment(const std::string& mode, long long N, int threads) {
    auto start_time = std::chrono::high_resolution_clock::now();
    double pi_approx = 0.0;

    if (mode == "seq") {
        pi_approx = sequential_pi(N);
    } else if (mode == "omp") {
        pi_approx = omp_pi(N, threads);
    } else if (mode == "std") {
        pi_approx = std_thread_pi(N, threads);
    } else if (mode == "pthread") {
        pi_approx = pthread_pi(N, threads);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return {"", 0, 0, 0.0, 0.0, 0.0, 0.0};
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double error = std::abs(pi_approx - PI_REF);

    return {mode, N, threads, elapsed.count(), pi_approx, error, 0.0};
}

void run_benchmark_suite() {
    const std::vector<long long> Ns = {1000000, 10000000, 100000000};
    const std::vector<int> Threads = {1, 2, 4, 8, 12, 16};
    const std::vector<std::string> Modes = {"seq", "omp", "std", "pthread"};
    const std::string OutputCSV = "benchmark_results_cpp.csv";

    std::vector<RunResult> results;
    std::map<long long, double> baseline_times;

    std::ofstream csv_file(OutputCSV);
    csv_file << "Mode,N,Threads,Time,Pi_Approx,Error,Speedup" << std::endl;

    std::cout << "Starting Benchmark Suite..." << std::endl;

    for (long long n : Ns) {
        std::cout << "\nRunning benchmarks for N = " << n << std::endl;

        // Run Sequential
        std::cout << "  Running Sequential..." << std::endl;
        RunResult seq_res = run_experiment("seq", n, 1);
        seq_res.speedup = 1.0;
        baseline_times[n] = seq_res.time;
        results.push_back(seq_res);

        csv_file << seq_res.mode << "," << seq_res.n << "," << seq_res.threads << "," 
                 << std::fixed << std::setprecision(9) << seq_res.time << "," 
                 << std::setprecision(15) << seq_res.pi_approx << "," 
                 << std::scientific << std::setprecision(9) << seq_res.error << ","
                 << std::fixed << std::setprecision(4) << seq_res.speedup << std::endl;

        // Run Parallel
        for (const auto& mode : Modes) {
            if (mode == "seq") continue;

            for (int t : Threads) {
                std::cout << "  Running " << mode << " with " << t << " threads..." << std::endl;
                RunResult res = run_experiment(mode, n, t);
                
                if (baseline_times.count(n)) {
                    res.speedup = baseline_times[n] / res.time;
                } else {
                    res.speedup = 0.0;
                }
                results.push_back(res);

                csv_file << res.mode << "," << res.n << "," << res.threads << "," 
                         << std::fixed << std::setprecision(9) << res.time << "," 
                         << std::setprecision(15) << res.pi_approx << "," 
                         << std::scientific << std::setprecision(9) << res.error << ","
                         << std::fixed << std::setprecision(4) << res.speedup << std::endl;
            }
        }
    }

    csv_file.close();
    std::cout << "\nResults saved to " << OutputCSV << std::endl;

    // Amdahl's Law Analysis
    std::cout << "\n--- Amdahl's Law Analysis ---" << std::endl;
    std::map<long long, std::map<std::string, std::pair<int, double>>> data;

    for (const auto& r : results) {
        if (r.mode == "seq") continue;
        auto& entry = data[r.n][r.mode];
        if (r.speedup > entry.second) {
            entry = {r.threads, r.speedup};
        }
    }

    for (const auto& n_entry : data) {
        long long n = n_entry.first;
        std::cout << "\nN = " << n << ":" << std::endl;
        for (const auto& mode_entry : n_entry.second) {
            std::string mode = mode_entry.first;
            int P = mode_entry.second.first;
            double Sp = mode_entry.second.second;

            if (P > 1 && Sp > 0) {
                double f = (1.0 - Sp/P) / (Sp * (1.0 - 1.0/P));
                std::cout << "  " << mode << ": Max Speedup " << std::fixed << std::setprecision(2) << Sp 
                          << " at P=" << P << ". Estimated Serial Fraction f = " 
                          << std::setprecision(4) << f << " (" << std::setprecision(2) << f*100 << "%)" << std::endl;
            } else {
                std::cout << "  " << mode << ": Insufficient data for Amdahl analysis" << std::endl;
            }
        }
    }
}

// ---------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------

int main(int argc, char* argv[]) {
    // If no arguments, run benchmark suite by default
    if (argc < 2) {
        std::cout << "No arguments provided. Running full benchmark suite by default..." << std::endl;
        std::cout << "To run a single instance, use: " << argv[0] << " <mode> <N> [threads]" << std::endl;
        run_benchmark_suite();
        return 0;
    }

    // Check if user explicitly asked for benchmark
    if (std::string(argv[1]) == "benchmark") {
        run_benchmark_suite();
        return 0;
    }

    // Otherwise, parse arguments for single run
    if (argc < 3) {
         std::cerr << "Usage: " << argv[0] << " <mode> <N> [threads]" << std::endl;
         std::cerr << "       " << argv[0] << " (runs benchmark)" << std::endl;
         std::cerr << "Modes: seq, omp, std, pthread" << std::endl;
         return 1;
    }

    std::string mode = argv[1];
    long long N = std::stoll(argv[2]);
    int threads = 1;
    if (argc >= 4) {
        threads = std::stoi(argv[3]);
    }

    RunResult res = run_experiment(mode, N, threads);

    // Output format: Mode, N, Threads, Time(s), Pi_Approx, Error
    std::cout << res.mode << "," << res.n << "," << res.threads << "," 
              << std::fixed << std::setprecision(9) << res.time << "," 
              << std::setprecision(15) << res.pi_approx << "," 
              << std::scientific << std::setprecision(9) << res.error << std::endl;

    return 0;
}
