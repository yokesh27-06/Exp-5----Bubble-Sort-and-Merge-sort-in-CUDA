# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```
!apt-get install -y nvidia-cuda-toolkit
```
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```
```
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

// Kernel for Bubble Sort
__global__ void bubbleSortKernel(int *d_arr, int n) {
    int temp;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure we're using a single block for Bubble Sort
    if (blockIdx.x > 0) return;  // Only use the first block

    // Perform bubble sort across multiple passes
    for (int i = 0; i < n - 1; i++) {
        if (idx < n - 1 - i) {
            if (d_arr[idx] > d_arr[idx + 1]) {
                // Swap
                temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }
        __syncthreads(); // Synchronize threads after each pass
    }
}
// Device function for merging arrays
__device__ void merge(int *arr, int left, int mid, int right, int *temp) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
}

// Kernel for Merge Sort
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Loop over merging sizes: 1, 2, 4, 8, ...
    for (int size = 1; size < n; size *= 2) {
        int left = 2 * size * tid;
        if (left < n) {
            int mid = min(left + size - 1, n - 1);
            int right = min(left + 2 * size - 1, n - 1);

            merge(d_arr, left, mid, right, d_temp);  // Each thread merges one part
        }

        // Sync threads in the block to ensure all merges at this level are done
        __syncthreads();

        // Swap the array pointers
        if (tid == 0) {
            int *temp = d_arr;
            d_arr = d_temp;
            d_temp = temp;
        }

        // Sync again before starting next merge size
        __syncthreads();
    }
}

// Host function for merging arrays
void mergeHost(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// Bubble Sort on GPU
void bubbleSort(int *arr, int n, int blockSize, int numBlocks) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bubbleSortKernel<<<numBlocks, blockSize>>>(d_arr, n);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);
}

// Merge Sort on GPU
void mergeSort(int *arr, int n, int blockSize, int numBlocks) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Start GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    mergeSortKernel<<<numBlocks, blockSize>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize(); // Wait for GPU to finish

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

// Bubble Sort on CPU
void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

// Merge Sort on CPU
void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1);

            mergeHost(arr, left, mid, right); // Call host merge
            left += 2 * size;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

// Main function
int main() {
    int n_array[] = {500, 1000};
    for (int i = 0; i < 2; i++) {
        int n = n_array[i];
        int *arr = (int*)malloc(n * sizeof(int));

        int blockSize_array[] = {16, 32};

        for (int i = 0; i < 2; i++) {

            int blockSize = blockSize_array[i]; // or higher, depending on the architecture
            int numBlocks = (n + blockSize - 1) / blockSize;

            printf("\nArray Size:%d\nBlock Size:%d\nNum Blocks:%d\n", n, blockSize, numBlocks);

            // Generating random array
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }

            // Bubble Sort CPU
            bubbleSortCPU(arr, n);

            // Generating random array again for GPU
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }

            // Bubble Sort GPU
            bubbleSort(arr, n, blockSize, numBlocks);

            // Generating random array again for GPU
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }

            // Merge Sort CPU
            mergeSortCPU(arr, n);
            // Generating random array again for Merge Sort
            for (int i = 0; i < n; i++) {
                arr[i] = rand() % 1000;
            }
            // Merge Sort GPU
            mergeSort(arr, n, blockSize, numBlocks);
            printf("\n");
        }
        free(arr);
    }
    return 0;
}

```
## OUTPUT:
<img width="569" height="838" alt="image" src="https://github.com/user-attachments/assets/5f229ef7-b37b-4ca9-9fca-48b4d298de66" />

## RESULT:
Thus, the program has been executed using CUDA to accelerate sorting operations and compare the performance of CPU and GPU implementations of Bubble Sort and Merge Sort.
