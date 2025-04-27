#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

// cofig file, make changes here
#include "config.h"
#include "utils.h"

// Add CUDA error checker
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }


/*
    ### Original version ###
    This is a 1D kernel.
    Only supports computing distances to one test point at a time.
    Only use global memory, which is very slow.

    ### Current version ###
    This kernel is a tiled GPU implementation that optimizes distance computation between all test 
    and training points using shared memory and 2D thread indexing.
    Each thread computes the distance between one training and one test point.
    Use shared memory to get high performance
*/
__global__ void batchCalcDistance (float *X_train, float *X_test, float *distance)
{
    /*
        Use shared memory to speed up repeated memory accesses
        Epecifically reduce global memory reads. Global memory much slower than shared memory.
    */

    // shared by threads in x-direction
    __shared__ float tile_train[BLOCK_X][NFEATURES];
    // shared by threads in y-direction
    __shared__ float tile_test[BLOCK_Y][NFEATURES];   

    // Fully tiled
    int trainIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int testIdx  = blockIdx.y * blockDim.y + threadIdx.y;

    // Avoiding redundant loads
    if (trainIdx < NTRAIN && threadIdx.y == 0) {
        for (int i = 0; i < NFEATURES; i++) {
            tile_train[threadIdx.x][i] = X_train[trainIdx * NFEATURES + i];
        }
    }
    if (testIdx < NTEST && threadIdx.x == 0) {
        for (int i = 0; i < NFEATURES; i++) {
            tile_test[threadIdx.y][i] = X_test[testIdx * NFEATURES + i];
        }
    }

    // Synchronize threads because of the use of shared memory
    __syncthreads();

    // Calculate distances
    if (trainIdx < NTRAIN && testIdx < NTEST) {
        float dist = 0.0f;
        for (int i = 0; i < NFEATURES; ++i) {
            float diff = tile_train[threadIdx.x][i] - tile_test[threadIdx.y][i];
            dist += diff * diff;
        }
        distance[testIdx * NTRAIN + trainIdx] = dist;
    }
}

/*
    ### Original version ###
    Sorts all NTRAIN distances for each test sample, which need O(N * log (N)) per test sample. 
    This means we need O(NTEST * N * log(N)), very slow.

    ### Current version ###
    Maintains a heap of K smallest distances
    O(N * K) per test sample, which is O(NTEST * N * K)
    Much faster when K < NTRAIN, especially for high NTRAIN. 
    (Usually the number of training data will much higher than K)

    Caution: This function only support >= 32 threads and < 1024 threads to run
    So we test 32 to 512 threads per block
*/
__global__ void findKMin(float *distances, int *min_indexes) {
    // Compute global thread index for training points
    int trainIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Which test instance this block handles
    int testIdx = blockIdx.y;

    // Dynamic shared memory
    extern __shared__ float sdata[];

    // First part of shared memory holds indexes, second part holds distances
    int *sharedIndexes = (int*) sdata;
    float *sharedDistances = (float*) (sharedIndexes + (K * blockDim.x));

    // Each thread initializes its own top-K buffer in shared memory
    if (trainIdx < NTRAIN) {
        for (int k = 0; k < K; k++) {
            int pos = k * blockDim.x + threadIdx.x;
            sharedIndexes[pos] = -1;
            sharedDistances[pos] = FLT_MAX;
        }
    }

    // Make sure all threads have finished initialization
    __syncthreads();

    // Each thread scans through the distances for the certain test instance
    // in strides of blockDim.x, covering all training points
    for (int i = trainIdx; i < NTRAIN; i += blockDim.x) {
        float d = distances[testIdx * NTRAIN + i];
        int idx = i;

        // Insert (d, idx) into the local sorted K-list (largest to smallest)
        for (int j = K - 1; j >= 0; j--) {
            int pos = j * blockDim.x + threadIdx.x;
            if (sharedDistances[pos] >= d) {
                // If currently at end of list, just replace
                if (j == K - 1) {
                    sharedDistances[pos] = d;
                    sharedIndexes[pos] = idx;
                } else {
                    // Shift bigger entries down by one
                    for (int l = K - 1; l > j; l--) {
                        int to = l * blockDim.x + threadIdx.x;
                        int from = (l - 1) * blockDim.x + threadIdx.x;
                        sharedDistances[to] = sharedDistances[from];
                        sharedIndexes[to] = sharedIndexes[from];
                    }
                    // Insert new entry
                    sharedDistances[pos] = d;
                    sharedIndexes[pos] = idx;
                }
            }
        }
    }

    // Wait for all threads to finish insertion
    __syncthreads();

    // Merge heaps in groups of 16 threads: threadIdx.x % 16 == 0 is a subgroup leader
    if (threadIdx.x % 16 == 0) {
        for (int t = 0; t < 16; t++) {
            int other = threadIdx.x + t;
            for (int j = K - 1; j >= 0; j--) {
                // Compute the shared-memory positions
                int leaderPos = j * blockDim.x + threadIdx.x;
                int otherPos = j * blockDim.x + other;
                if (sharedDistances[leaderPos] >= sharedDistances[otherPos]) {
                    // If at the end of the list, overwrite
                    if (j == K - 1) {
                        sharedDistances[leaderPos] = sharedDistances[otherPos];
                        sharedIndexes[leaderPos] = sharedIndexes[otherPos];
                    } else {
                        for (int l = K - 1; l > j; l--) {
                            int to = l * blockDim.x + threadIdx.x;
                            int from = (l - 1) * blockDim.x + threadIdx.x;
                            sharedDistances[to] = sharedDistances[from];
                            sharedIndexes[to] = sharedIndexes[from];
                        }

                        // Insert the new smaller distance at position j
                        sharedDistances[leaderPos] = sharedDistances[otherPos];
                        sharedIndexes[leaderPos] = sharedIndexes[otherPos];
                    }
                }
            }
        }
    }

    // Wait again before final merge
    __syncthreads();

    // Final merge by thread 0: combine all subgroup leaders into final top-K
    if (threadIdx.x == 0) {
        int numLeaders = blockDim.x / 16;
        // Iterate over each subgroup leader
        for (int g = 0; g < numLeaders; g++) {
            // index of that leader
            int other = g * 16;
            for (int j = K - 1; j >= 0; j--) {
                // j-th slot of thread 0
                int mainPos = j * blockDim.x;

                // j-th slot of the other leader
                int otherPos = j * blockDim.x + other;

                if (sharedDistances[mainPos] >= sharedDistances[otherPos]) {
                    if (j == K - 1) {
                        // Overwrite
                        sharedDistances[mainPos] = sharedDistances[otherPos];
                        sharedIndexes[mainPos] = sharedIndexes[otherPos];
                    } else {
                        // Insert at position j
                        for (int l = K - 1; l > j; l--) {
                            sharedDistances[l * blockDim.x] = sharedDistances[(l - 1) * blockDim.x];
                            sharedIndexes[l * blockDim.x] = sharedIndexes[(l - 1) * blockDim.x];
                        }
                        sharedDistances[mainPos] = sharedDistances[otherPos];
                        sharedIndexes[mainPos] = sharedIndexes[otherPos];
                    }
                }
            }
        }

        // Write the final K nearest neighbor indices back to global memory
        for (int k = 0; k < K; k++) {
            min_indexes[testIdx * K + k] = sharedIndexes[k * blockDim.x];
        }
    }
}

int predict(int *indexes, float *y_train)
{
    float* neighborCount = getFloatMat(NCLASSES, 1);
    float* probability = getFloatMat(NCLASSES, 1);

    for (int i = 0; i < NCLASSES; i++)
        neighborCount[i] = 0;

    for (int i = 0; i < K; i++) {
        int train_idx = indexes[i];
        if (train_idx < 0 || train_idx >= NTRAIN) {
            printf("Invalid index: %d (i=%d)\n", train_idx, i);
            exit(1);
        }
        int label = (int)y_train[train_idx];
        neighborCount[label]++;
    }

    for (int i = 0; i < NCLASSES; i++)
        probability[i] = neighborCount[i] / (float)K;

    int predicted_class = (int)getMax(neighborCount, NCLASSES);

    free(neighborCount);
    free(probability);

    return predicted_class;
}


int *fit(float *X_train, float *y_train, float *X_test,
    float *X_traind, float *y_traind, float *X_testd,
    float *distanced, int *min_indexes, int *min_indexesd)
{

    // Create timer event
    cudaEvent_t st1, et1, st2, et2;
    float time1, time2;

    cudaEventCreate(&st1);
    cudaEventCreate(&et1);
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);
    
    // Should match the whole batch of distance between test data and train data
    float *distance = getFloatMat(NTEST, NTRAIN);

    int X_train_size = sizeof(float)*NFEATURES*NTRAIN;
    int y_train_size = sizeof(float)*NTRAIN;
    int X_test_size = sizeof(float)*NFEATURES*NTEST;
    int distance_size = sizeof(float)*NTEST*NTRAIN;
    
    cudaMemcpy(X_traind, X_train, X_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_traind, y_train, y_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(X_testd, X_test, X_test_size, cudaMemcpyHostToDevice);
   
    // Number of threads in each block. 2D: BLOCK_X * BLOCK_Y
    dim3 block(BLOCK_X, BLOCK_Y);

    /*
        Number of blocks in each grid 
        Want to cover all NTRAIN * NTEST combinations using a 2D grid of blocks, where each block contains:
        BLOCK_X threads along the x-axis & BLOCK_Y threads along the y-axis
        Use (+ BLOCK_X - 1) is because we want to avoid missing data if NTRAIN isn't an exact multiple of BLOCK_X.
    */
    dim3 grid((NTRAIN + BLOCK_X - 1) / BLOCK_X, (NTEST + BLOCK_Y - 1) / BLOCK_Y);

    // Start record
    cudaEventRecord(st1);

    

    /*
        Launch distance kernel 
        Use batch distance calculation
        Use 2D launch
    */
    batchCalcDistance<<<grid, block>>>(X_traind, X_testd, distanced);

    // Check CUDA
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaDeviceSynchronize();

    // End record
    cudaEventRecord(et1);
    cudaEventSynchronize(et1);
    cudaEventElapsedTime(&time1, st1, et1);
    
    cudaMemcpy(distance, distanced, distance_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(distanced, distance, distance_size, cudaMemcpyHostToDevice);

    /*
        Use one block per test point
        We want 1 thread block to be responsible for finding the top-K distances in that row
        This single block will collaborate via shared memory to process the entire row efficiently
    */
    dim3 gridFindKMin(1, NTEST);  


    // threads per block
    dim3 blockFindKMin(BLOCK_X * BLOCK_Y);
    
    /*
        Setup dynamic shared memory
        The kernel uses shared memory to store two arrays for each thread:
        1. distances: Each thread store K float values
        2. indexes: Each thread store K int values
        So we need (BLOCK_X * BLOCK_Y) * K * sizeof(float) + (BLOCK_X * BLOCK_Y) * K * sizeof(int)
    */
    size_t shared_mem_size = (BLOCK_X * BLOCK_Y) * K * sizeof(float) + (BLOCK_X * BLOCK_Y) * K * sizeof(int);

    
    // Start record
    cudaEventRecord(st2);

    // Call sorting kernel
    findKMin<<<gridFindKMin, blockFindKMin, shared_mem_size>>>(distanced, min_indexesd);

    // Check CUDA
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // End record
    cudaEventRecord(et2);
    cudaEventSynchronize(et2);
    cudaEventElapsedTime(&time2, st2, et2);
    
    // min_indexes stores the indices into the training set for the K nearest neighbors
    cudaMemcpy(min_indexes, min_indexesd, K * NTEST * sizeof(int), cudaMemcpyDeviceToHost);
    
    free(distance);

    printf("\nkernel batchCalcDistance: %.6f ms | kernel findKMin: %.6f ms\n", time1, time2);
    
    return min_indexes;
}

void readData(float **X_train, float **y_train, float **X_test, float **y_test)
{
    *X_train = initFeatures(X_TRAIN_PATH);
	*y_train = initLabels(Y_TRAIN_PATH);

	*X_test = initFeatures(X_TEST_PATH);
	*y_test = initLabels(Y_TEST_PATH);
}

int knn(float *X_train, float *y_train, float *X_test,
    float *X_traind, float *y_traind, float *X_testd,
    float *distanced, int *min_indexes, int *min_indexesd)
{

    /*
        Directly return the indexes of predictions
    */
    int *indexes = fit(X_train, y_train, X_test,
                        X_traind, y_traind, X_testd,
                        distanced, min_indexes, min_indexesd);

    int predicted_class = predict(indexes, y_train);
    free(indexes);
    return predicted_class;
}

int main()
{
    float *X_train, *y_train, *X_test, *y_test, et;
    float *X_traind, *y_traind, *X_testd, *distanced;
    int *min_indexes, *min_indexesd;

    min_indexes = (int *)calloc(NTEST * K, sizeof(int));

    // Move all memory allocation operations outside of the knn fit function
    cudaMalloc((void**)&X_traind, sizeof(float)*NFEATURES*NTRAIN);
    cudaMalloc((void**)&y_traind, sizeof(float)*NTRAIN);
    cudaMalloc((void**)&X_testd, sizeof(float)*NFEATURES*NTEST);
    cudaMalloc((void**)&distanced, sizeof(float)*NTRAIN*NTEST);
    cudaMalloc((void**)&min_indexesd, sizeof(int) * NTEST * K);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
 
    
    //read data
    readData(&X_train, &y_train, &X_test, &y_test);
    
    //call knn
    int predicted_class = knn(X_train, y_train, X_test,
        X_traind, y_traind, X_testd,
        distanced, min_indexes, min_indexesd);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
   
    printf("Time taken: %fms\n", et);
    
    
    // printf("Predicted label: %d True label: %d\n", predicted_class, (int)y_test[randId]);
    // Free the allocated memory
    cudaFree(X_traind);
    cudaFree(y_traind);
    cudaFree(X_testd);
    cudaFree(distanced);
    cudaFree(min_indexesd);
     
	free(X_train);
	free(y_train);

	free(X_test);
	free(y_test);
    
    return 0;
}
