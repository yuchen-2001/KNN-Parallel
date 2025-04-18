#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// cofig file, make changes here
#include "config.h"
#include "utils.h"

// Add CUDA error checker
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }



// Instead of calculate the distance between single test data point and all train data points,
// this kernel is a tiled GPU implementation that optimizes distance computation between all test 
// and training points using shared memory and 2D thread indexing.
__global__ void batchCalcDistance (float *X_train, float *X_test, float *distance)
{
    // Tiling with Shared Memory
    // shared by threads in x-direction
    __shared__ float tile_train[BLOCK_X][NFEATURES];
    // shared by threads in y-direction
    __shared__ float tile_test[BLOCK_Y][NFEATURES];   

    // Fully tiled
    int train_id = blockIdx.x * blockDim.x + threadIdx.x;
    int test_id  = blockIdx.y * blockDim.y + threadIdx.y;

    // Avoiding redundant loads
    if (train_id < NTRAIN && threadIdx.y == 0) {
        for (int i = 0; i < NFEATURES; i++) {
            tile_train[threadIdx.x][i] = X_train[train_id * NFEATURES + i];
        }
    }
    if (test_id < NTEST && threadIdx.x == 0) {
        for (int i = 0; i < NFEATURES; i++) {
            tile_test[threadIdx.y][i] = X_test[test_id * NFEATURES + i];
        }
    }

    // Synchronize threads because of the use of shared memory
    __syncthreads();

    // Calculate distances
    if (train_id < NTRAIN && test_id < NTEST) {
        float dist = 0.0f;
        for (int i = 0; i < NFEATURES; ++i) {
            float diff = tile_train[threadIdx.x][i] - tile_test[threadIdx.y][i];
            dist += diff * diff;
        }
        distance[test_id * NTRAIN + train_id] = dist;
    }
}

__global__ void sortArray2D(float *distance, float *ytrain, float *sortedDistance, float *sortedYtrain) {
    int train_id = blockIdx.x * blockDim.x + threadIdx.x; // column
    int test_id  = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (test_id >= NTEST || train_id >= NTRAIN) return;

    float element = distance[test_id * NTRAIN + train_id];
    float label = ytrain[train_id];
    int position = 0;

    for (int j = 0; j < NTRAIN; ++j) {
        float other = distance[test_id * NTRAIN + j];
        if (other < element || (other == element && train_id < j)) {
            position++;
        }
    }

    sortedDistance[test_id * NTRAIN + position] = element;
    sortedYtrain[test_id * NTRAIN + position] = label;
}

int predict(float *labels)
{
	float* neighborCount = getFloatMat(NCLASSES, 1);
    
	float* probability = getFloatMat(NCLASSES, 1);

	int i;
    for(i=0; i<NCLASSES; i++)
        neighborCount[i] = 0;

	for(i=0; i<K; i++)
		neighborCount[(int)labels[i]]++;

	for(i=0; i<NCLASSES; i++)
		probability[i] = neighborCount[i]*1.0/(float)K*1.0;
	
	int predicted_class = (int)getMax(neighborCount, NCLASSES);

	// for(i=0; i<TOPN; i++)
	// 	printf(" %s: %f ", classes[i], probability[i]);

	free(neighborCount);
	free(probability);

	return predicted_class;
}

float *fit(float *X_train, float *y_train, float *X_test,
    float *X_traind, float *y_traind, float *X_testd,
    float *distanced, float *sortedDistanced, float *sortedytraind)
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
    float *sortedytrain = getFloatMat(NTEST, NTRAIN);

    int X_train_size = sizeof(float)*NFEATURES*NTRAIN;
    int y_train_size = sizeof(float)*NTRAIN;
    int X_test_size = sizeof(float)*NFEATURES*NTEST;
    int distance_size = sizeof(float)*NTEST*NTRAIN;
    
    cudaMemcpy(X_traind, X_train, X_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_traind, y_train, y_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(X_testd, X_test, X_test_size, cudaMemcpyHostToDevice);
   
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((NTRAIN + BLOCK_X - 1) / BLOCK_X, (NTEST + BLOCK_Y - 1) / BLOCK_Y);

    // Start record
    cudaEventRecord(st1);

    //TODO: launch distance kernel 
    // Use batch distance calcultion
    // Use 2D launch
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
    
    // Start record
    cudaEventRecord(st2);

    //TODO: call sorting kernel
    sortArray2D <<< grid, block >>> (distanced, y_traind, sortedDistanced, sortedytraind);

    // Check CUDA
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // End record
    cudaEventRecord(et2);
    cudaEventSynchronize(et2);
    cudaEventElapsedTime(&time2, st2, et2);
    
    // We do not need sortedDistance because we only need to use sortedDistance
    // cudaMemcpy(sortedDistance, sortedDistanced, distance_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(sortedytrain, sortedytraind, y_train_size, cudaMemcpyDeviceToHost);
    
    free(distance);

    // printf("\nkernel calcDistance: %.6f ms | kernel sortArray: %.6f ms\n", time1, time2);
    
    return sortedytrain;
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
    float *distanced, float *sortedDistanced, float *sortedytraind)
{
    float *labels = fit(X_train, y_train, X_test,
                        X_traind, y_traind, X_testd,
                        distanced, sortedDistanced, sortedytraind);

    int predicted_class = predict(labels);
    free(labels);
    return predicted_class;
}

int main()
{
    float *X_train, *y_train, *X_test, *y_test, et;
    float *X_traind, *y_traind, *X_testd, *distanced;
    float *sortedDistanced, *sortedytraind;
    
    // Move all memory allocation operations outside of the knn fit function
    cudaMalloc((void**)&X_traind, sizeof(float)*NFEATURES*NTRAIN);
    cudaMalloc((void**)&y_traind, sizeof(float)*NTRAIN);
    cudaMalloc((void**)&X_testd, sizeof(float)*NFEATURES*NTEST);
    cudaMalloc((void**)&distanced, sizeof(float)*NTRAIN*NTEST);
    cudaMalloc((void**)&sortedDistanced, sizeof(float)*NTRAIN*NTEST);
    cudaMalloc((void**)&sortedytraind, sizeof(float)*NTRAIN*NTEST);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
 
    
    //read data
    readData(&X_train, &y_train, &X_test, &y_test);
    
    //call knn
    int predicted_class = knn(X_train, y_train, X_test,
        X_traind, y_traind, X_testd,
        distanced, sortedDistanced, sortedytraind);

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
    cudaFree(sortedDistanced);
    cudaFree(sortedytraind);
     
	free(X_train);
	free(y_train);

	free(X_test);
	free(y_test);
    
    return 0;
}
