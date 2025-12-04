#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace std::chrono;

// Macro para chequear errores de CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CPU BFS
vector<int> bfs_cpu(const vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<int> dist(n, -1);
    queue<int> q;

    dist[start] = 0;
    q.push(start);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : graph[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}

// GPU BFS Kernel
__global__ void bfs_kernel(const int* row_ptr, const int* col_idx,
                           int* dist, int* frontier, int* next_frontier,
                           const int* frontier_size, int* next_frontier_size,
                           int current_level) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Solo trabajamos si el thread esta dentro del rango de la frontera actual
    if (tid < *frontier_size) {
        int u = frontier[tid];
        int start_edge = row_ptr[u];
        int end_edge = row_ptr[u + 1];

        for (int i = start_edge; i < end_edge; i++) {
            int v = col_idx[i];

            // Si el nodo no ha sido visitado (-1), intentamos visitarlo
            if (atomicCAS(&dist[v], -1, current_level) == -1) {
                // Reservamos posición en la próxima frontera
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}

// Funcion auxiliar para calcular memoria usada
double get_memory_mb(int n, int e) {
    size_t total_bytes = 0;
    total_bytes += (n + 1) * sizeof(int); // row_ptr
    total_bytes += e * sizeof(int);       // col_idx
    total_bytes += n * sizeof(int);       // dist
    total_bytes += n * sizeof(int);       // frontier
    total_bytes += n * sizeof(int);       // next_frontier
    total_bytes += 2 * sizeof(int);       // sizes
    return total_bytes / (1024.0 * 1024.0);
}

//  GPU BFS Wrapper
vector<int> bfs_gpu(const vector<vector<int>>& graph, int start, double &mem_mb) {
    int n = graph.size();

    // Convertir a CSR
    vector<int> row_ptr(n + 1, 0), col_idx;
    for (int i = 0; i < n; i++) {
        row_ptr[i + 1] = row_ptr[i] + graph[i].size();
        for (int v : graph[i]) {
            col_idx.push_back(v);
        }
    }
    int total_edges = col_idx.size();
    mem_mb = get_memory_mb(n, total_edges);

    // Alloc GPU
    int *d_row_ptr, *d_col_idx, *d_dist;
    int *d_frontier, *d_next_frontier;
    int *d_frontier_size, *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, total_edges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier_size, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

    // 3. Inicializar y Copiar
    vector<int> dist(n, -1);
    dist[start] = 0;

    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(), total_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int frontier_size = 1;
    int next_frontier_size = 0;
    vector<int> initial_frontier = {start};

    CUDA_CHECK(cudaMemcpy(d_frontier, initial_frontier.data(), sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_size, &frontier_size, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_frontier_size, &next_frontier_size, sizeof(int), cudaMemcpyHostToDevice));

    int level = 1;
    int block_size = 256;

    // Loop BFS
    while (frontier_size > 0) {
        int grid_size = (frontier_size + block_size - 1) / block_size;

        bfs_kernel<<<grid_size, block_size>>>(
            d_row_ptr, d_col_idx, d_dist,
            d_frontier, d_next_frontier,
            d_frontier_size, d_next_frontier_size,
            level
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap pointers
        swap(d_frontier, d_next_frontier);
        swap(d_frontier_size, d_next_frontier_size);

        // Leer nuevo tamaño
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));

        // Resetear el contador de la "proxima" frontera
        next_frontier_size = 0;
        CUDA_CHECK(cudaMemcpy(d_next_frontier_size, &next_frontier_size, sizeof(int), cudaMemcpyHostToDevice));

        level++;
    }

    // Copiar resultados
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_dist);
    cudaFree(d_frontier); cudaFree(d_next_frontier);
    cudaFree(d_frontier_size); cudaFree(d_next_frontier_size);

    return dist;
}

// Funciones Auxiliares
vector<vector<int>> generate_graph(int n, int avg_degree) {
    vector<vector<int>> graph(n);
    random_device rd;
    mt19937 gen(42); // Seed fija
    uniform_int_distribution<> dis(0, n - 1);

    // Generar aristas aleatorias
    long long target_edges = (long long)n * avg_degree / 2;
    for (long long i = 0; i < target_edges; i++) {
        int u = dis(gen);
        int v = dis(gen);
        if (u != v) {
            graph[u].push_back(v);
            graph[v].push_back(u);
        }
    }
    return graph;
}

int count_edges(const vector<vector<int>>& graph) {
    int edges = 0;
    for (const auto& adj : graph) edges += adj.size();
    return edges; // Aristas dirigidas totales en lista de adyacencia
}

int main() {
    cudaFree(0);
    ofstream csv_file("bfs_results.csv");
    csv_file << "Size_V,Edges_E,CPU_Time_ms,GPU_Time_ms,Speedup\n";

    vector<int> sizes = {1000, 5000, 10000, 50000, 100000, 200000};
    int avg_degree = 10;

    cout << "Running BFS Tests and writing to bfs_results.csv...\n";
    cout << "V\tCPU(ms)\tGPU(ms)\tSpeedup\n";

    for (int n : sizes) {
        auto graph = generate_graph(n, avg_degree);
        int edges = count_edges(graph);

        // CPU
        auto start = high_resolution_clock::now();
        bfs_cpu(graph, 0);
        auto end = high_resolution_clock::now();
        double cpu_ms = duration_cast<microseconds>(end - start).count() / 1000.0;

        // GPU
        double mem_mb;
        start = high_resolution_clock::now();
        bfs_gpu(graph, 0, mem_mb);
        end = high_resolution_clock::now();
        double gpu_ms = duration_cast<microseconds>(end - start).count() / 1000.0;

        double speedup = cpu_ms / max(0.001, gpu_ms);

        // Output consola
        cout << n << "\t" << cpu_ms << "\t" << gpu_ms << "\t" << fixed << setprecision(2) << speedup << "x" << endl;

        // Output CSV
        csv_file << n << "," << edges << "," << cpu_ms << "," << gpu_ms << "," << speedup << "\n";
    }

    csv_file.close();
    cout << "\nDone! 'bfs_results.csv' generated.\n";
    return 0;
}


'''
# 1. Compilar forzando la arquitectura de la T4 (sm_75)
!nvcc -arch=sm_75 bfs_cuda.cu -o bfs_cuda

# 2. Ejecutar
!./bfs_cuda

'''

