#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define BLOCK_SIZE 256

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void fill_vector(double* vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (double)(rand() % 100) / 10.0;
    }
}

double sum_cpu(double* vec, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += vec[i];
    }
    return sum;
}

__global__ void sum_reduction_kernel(double* input, double* partial_sums, int n) {
    extern __shared__ double shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0.0;
    }
    
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_sums[blockIdx.x] = shared_data[0];
    }
}

int main() {
    int sizes[] = {1000, 10000, 100000, 500000, 1000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("========================================================================\n");
    printf("Сравнение CPU vs GPU: Суммирование вектора (double)\n");
    printf("========================================================================\n");
    printf("%10s | %12s | %12s | %10s | %12s\n", 
           "Размер", "CPU (сек)", "GPU (сек)", "Ускорение", "Разница");
    printf("------------------------------------------------------------------------\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(double);
        
        double* h_vec = (double*)malloc(bytes);
        double* h_partial_sums = NULL;
        
        srand(42);
        fill_vector(h_vec, n);
        
        double start = get_time();
        double cpu_sum = sum_cpu(h_vec, n);
        double cpu_time = get_time() - start;
        
        double *d_vec, *d_partial_sums;
        cudaMalloc(&d_vec, bytes);
        cudaMemcpy(d_vec, h_vec, bytes, cudaMemcpyHostToDevice);
        
        int block_size = BLOCK_SIZE;
        int num_blocks = (n + block_size - 1) / block_size;
        
        size_t partial_bytes = num_blocks * sizeof(double);
        cudaMalloc(&d_partial_sums, partial_bytes);
        h_partial_sums = (double*)malloc(partial_bytes);
        
        cudaEvent_t gpu_start, gpu_stop;
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_stop);
        
        cudaEventRecord(gpu_start);
        sum_reduction_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(d_vec, d_partial_sums, n);
        cudaMemcpy(h_partial_sums, d_partial_sums, partial_bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(gpu_stop);
        cudaEventSynchronize(gpu_stop);
        
        float gpu_time_ms;
        cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_stop);
        double gpu_time = gpu_time_ms / 1000.0;
        
        double gpu_sum = 0.0;
        for (int i = 0; i < num_blocks; i++) {
            gpu_sum += h_partial_sums[i];
        }
        
        double diff = fabs(cpu_sum - gpu_sum);
        
        printf("%10d | %12.6f | %12.6f | %10.2fx | %12.6e\n",
               n, cpu_time, gpu_time, cpu_time / gpu_time, diff);
        
        cudaFree(d_vec);
        cudaFree(d_partial_sums);
        free(h_vec);
        free(h_partial_sums);
    }
    
    printf("========================================================================\n");
    return 0;
}
