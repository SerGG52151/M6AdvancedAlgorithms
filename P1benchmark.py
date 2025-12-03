import subprocess
import os
import csv
import matplotlib.pyplot as plt
import platform

# Configuration
Ns = [10**6, 10**7, 10**8]
Threads = [1, 2, 4, 8, 12, 16]
Modes = ["seq", "omp", "std", "pthread"]
OutputCSV = "benchmark_results.csv"
Executable = "P1.exe" if platform.system() == "Windows" else "./P1"
SourceFile = "P1.cpp"

def compile_code():
    print("Compiling code...")
    # Ensure we link pthread. On Windows/MinGW it might be implicit or -lpthread
    cmd = ["g++", "-O3", "-fopenmp", SourceFile, "-o", "P1", "-lpthread"]
    try:
        subprocess.check_call(cmd)
        print("Compilation successful.")
        return True
    except subprocess.CalledProcessError:
        print("Compilation failed. Please ensure g++ is installed and supports -fopenmp and pthreads.")
        return False

def run_benchmark():
    results = []
    
    # Header for CSV
    header = ["Mode", "N", "Threads", "Time", "Pi_Approx", "Error", "Speedup"]
    results.append(header)

    # Baseline times for speedup calculation (Mode -> N -> Time)
    baseline_times = {}

    for N in Ns:
        print(f"\nRunning benchmarks for N = {N}")
        
        # Run Sequential first to get baseline
        print(f"  Running Sequential...")
        try:
            output = subprocess.check_output([Executable, "seq", str(N)], text=True).strip()
            parts = output.split(',')
            time_seq = float(parts[3])
            baseline_times[N] = time_seq
            results.append(parts + ["1.0"]) # Speedup is 1.0
        except subprocess.CalledProcessError as e:
            print(f"    Failed to run sequential: {e}")
            continue

        # Run Parallel versions
        for mode in ["omp", "std", "pthread"]:
            for t in Threads:
                print(f"  Running {mode} with {t} threads...")
                try:
                    output = subprocess.check_output([Executable, mode, str(N), str(t)], text=True).strip()
                    parts = output.split(',')
                    time_par = float(parts[3])
                    speedup = baseline_times[N] / time_par if N in baseline_times else 0.0
                    results.append(parts + [f"{speedup:.4f}"])
                except subprocess.CalledProcessError as e:
                    print(f"    Failed to run {mode} with {t} threads: {e}")

    return results

def save_csv(results):
    with open(OutputCSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    print(f"\nResults saved to {OutputCSV}")

def analyze_amdahl(results):
    print("\n--- Amdahl's Law Analysis ---")
    # Parse results to find max speedup for each mode/N
    # Structure: N -> Mode -> (MaxThreads, MaxSpeedup)
    data = {}
    
    for row in results[1:]:
        mode = row[0]
        if mode == "seq": continue
        n = int(row[1])
        threads = int(row[2])
        speedup = float(row[6])
        
        if n not in data: data[n] = {}
        if mode not in data[n]: data[n][mode] = (threads, speedup)
        
        # Keep the one with max threads (assuming scaling holds or saturates)
        # Or better, keep the max speedup found
        if speedup > data[n][mode][1]:
            data[n][mode] = (threads, speedup)

    # Calculate f
    # f = (1 - Sp/P) / (Sp * (1 - 1/P))
    for n in data:
        print(f"\nN = {n}:")
        for mode in data[n]:
            P, Sp = data[n][mode]
            if P > 1 and Sp > 0:
                try:
                    f = (1 - Sp/P) / (Sp * (1 - 1/P))
                    print(f"  {mode}: Max Speedup {Sp:.2f} at P={P}. Estimated Serial Fraction f = {f:.4f} ({f*100:.2f}%)")
                except ZeroDivisionError:
                    print(f"  {mode}: Error calculating f")
            else:
                print(f"  {mode}: Insufficient data for Amdahl analysis")

def plot_results(results):
    # Parse results for plotting
    # Structure: Mode -> N -> (Threads, Speedup)
    data = {}
    
    # Skip header
    for row in results[1:]:
        mode = row[0]
        n = int(row[1])
        threads = int(row[2])
        speedup = float(row[6])
        
        if mode == "seq":
            continue
            
        if n not in data:
            data[n] = {}
        if mode not in data[n]:
            data[n][mode] = {'threads': [], 'speedup': []}
            
        data[n][mode]['threads'].append(threads)
        data[n][mode]['speedup'].append(speedup)

    # Create plots
    for n in data:
        plt.figure(figsize=(10, 6))
        plt.title(f"Speedup vs Threads (N={n})")
        plt.xlabel("Threads")
        plt.ylabel("Speedup")
        plt.grid(True)
        plt.xticks(Threads)
        
        # Plot ideal speedup
        plt.plot(Threads, Threads, 'k--', label="Ideal Linear Speedup")
        
        for mode in data[n]:
            plt.plot(data[n][mode]['threads'], data[n][mode]['speedup'], marker='o', label=f"Mode: {mode}")
            
        plt.legend()
        plt.savefig(f"speedup_plot_N_{n}.png")
        print(f"Saved plot to speedup_plot_N_{n}.png")

if __name__ == "__main__":
    if compile_code():
        data = run_benchmark()
        save_csv(data)
        analyze_amdahl(data)
        try:
            plot_results(data)
        except Exception as e:
            print(f"Plotting failed (maybe matplotlib is missing?): {e}")
