#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// cofig file, make changes here
#include "config.h"

void checkFile(FILE *f)
{
	if (f == NULL)
	{
		printf("Error while reading file\n");
		exit(1);
	}
}


float *getFloatMat(int m, int n)
{
	float *mat = NULL;
	mat = (float*)malloc(m*n*sizeof(float));

	return mat;
}


float *initFeatures(const char path[])
{
	int index = 0;
	FILE *f  = NULL;
	float *mat = NULL;

	mat = getFloatMat(NTRAIN, NFEATURES);

	f = fopen(path, "r");
	checkFile(f);

	while (fscanf(f, "%f%*c", &mat[index]) == 1) //%*c ignores the comma while reading the CSV
		index++;

	fclose(f);
	return mat;
}

float getMax(float *x, int n)
{
	int i;
	float max = x[0];
	int maxIndex = 0;

	for(i=0; i<n; i++)
	{
		if (x[i] >= max)
		{
			max = x[i];
			maxIndex = i;
		}
	}

	return (float)maxIndex;
}

float *initLabels(const char path[])
{
	int index = 0;
	FILE *f  = NULL;
	float *mat = NULL;

	mat = getFloatMat(NTRAIN, 1);

	f = fopen(path, "r");
	checkFile(f);

	while (fscanf(f, "%f%*c", &mat[index]) == 1)
		index++;

	fclose(f);
	return mat;
}

// Instead of calculate the distance between single test data point and all train data points,
// directly calculate the whole batch of distances of test data points 
__global__ void batchCalcDistance (float *X_train, float *X_test, float *distance)
{
    // shared by threads in x-direction
    __shared__ float tile_train[BLOCK_X][NFEATURES];
    // shared by threads in y-direction
    __shared__ float tile_test[BLOCK_Y][NFEATURES];   

    // Fully tiled
    int train_id = blockIdx.x * blockDim.x + threadIdx.x;
    int test_id  = blockIdx.y * blockDim.y + threadIdx.y;

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

__global__ void sortArray (float *distance, float *ytrain, float *sortedDistance, float *sortedYtrain)
{
    int blockid = blockIdx.x;
    int threadid = blockDim.x*blockid + threadIdx.x;
    
    if (threadid < NTRAIN)
    {
        int i, position = 0;
        float element = distance[threadid];
        float label = ytrain[threadid];
    
        for(i=0; i<NTRAIN; i++)
        {
            if (distance[i] < element || (distance[i] == element && threadid < i) )
                position++;
        }
    
        sortedDistance[position] = element;
        sortedYtrain[position] = label;
    }
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

	for(i=0; i<TOPN; i++)
		printf(" %s: %f ", classes[i], probability[i]);

	free(neighborCount);
	free(probability);

	return predicted_class;
}

float *fit(float *X_train, float *y_train, float *X_test)
{
    float *X_traind, *y_traind, *X_testd, *distanced, *distance;

    // Create timer event
    cudaEvent_t st1;
    cudaEvent_t et1;
    cudaEvent_t st2;
    cudaEvent_t et2;

    float time1;
    float time2;

    cudaEventCreate(&st1);
    cudaEventCreate(&et1);
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);
    
    // Should match the whole batch of distance between test data and train data
    distance = getFloatMat(NTEST, NTRAIN);
    
    int X_train_size = sizeof(float)*NFEATURES*NTRAIN;
    int y_train_size = sizeof(float)*NTRAIN;
    int X_test_size = sizeof(float)*NFEATURES*NTEST;
    int distance_size = sizeof(float)*NTEST*NTRAIN;
    
    cudaMalloc((void**)&X_traind, X_train_size);
    cudaMalloc((void**)&y_traind, y_train_size);
    cudaMalloc((void**)&X_testd, X_test_size);
    cudaMalloc((void**)&distanced, distance_size);
    
    cudaMemcpy(X_traind, X_train, X_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_traind, y_train, y_train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(X_testd, X_test, X_test_size, cudaMemcpyHostToDevice);

    // Start record
    cudaEventRecord(st1);
   
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((NTRAIN + BLOCK_X - 1) / BLOCK_X, (NTEST + BLOCK_Y - 1) / BLOCK_Y);

    //TODO: launch distance kernel 
    // Use batch distance calcultion
    // Use 2D launch
    batchCalcDistance<<<grid, block>>>(X_traind, X_testd, distanced);
    
    cudaDeviceSynchronize();

    // End record
    cudaEventRecord(et1);
    cudaEventSynchronize(et1);
    cudaEventElapsedTime(&time1, st1, et1);
    
    cudaMemcpy(distance, distanced, distance_size, cudaMemcpyDeviceToHost);
    
    cudaFree(X_traind);
    cudaFree(X_testd);
    
    /* Launching a kernel to sort the distances and make corresponing swaps in the y_train */ 
    float *sortedDistance, *sortedDistanced, *sortedytrain, *sortedytraind;
       
    sortedDistance = getFloatMat(NTRAIN, 1);
    sortedytrain = getFloatMat(NTRAIN, 1);
    
    cudaMalloc((void**)&sortedDistanced, distance_size);
    cudaMalloc((void**)&sortedytraind, y_train_size);
    
    cudaMemcpy(distanced, distance, distance_size, cudaMemcpyHostToDevice);
    
    // Start record
    cudaEventRecord(st2);

    //TODO: call sorting kernel
    sortArray <<< NTRAIN/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (distanced, y_traind, sortedDistanced, sortedytraind);

    // End record
    cudaEventRecord(et2);
    cudaEventSynchronize(et2);
    cudaEventElapsedTime(&time2, st2, et2);
    
    cudaMemcpy(sortedDistance, sortedDistanced, distance_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(sortedytrain, sortedytraind, y_train_size, cudaMemcpyDeviceToHost);
    
    cudaFree(y_traind);
    cudaFree(distanced);
    cudaFree(sortedDistanced);
    cudaFree(sortedytraind);
    
    free(distance);
    free(sortedDistance);

    printf("\nkernel calcDistance: %.6f ms | kernel sortArray: %.6f ms\n", time1, time2);
    
    return sortedytrain;
}

void readData(float **X_train, float **y_train, float **X_test, float **y_test)
{
    *X_train = initFeatures(X_TRAIN_PATH);
	*y_train = initLabels(Y_TRAIN_PATH);

	*X_test = initFeatures(X_TEST_PATH);
	*y_test = initLabels(Y_TEST_PATH);
}

int knn(float *X_train, float *y_train, float *X_test)
{
    
    printf(" Fitting model ");
    
    float et;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
	float *labels = fit(X_train, y_train, X_test);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&et, start, stop);
   
    printf("Time taken: %fms", et);
    
    int predicted_class = predict(labels);
    return predicted_class;
}

int main()
{
    float *X_train;
	float *y_train;
	float *X_test;
	float *y_test;
 
    
    //read data
    readData(&X_train, &y_train, &X_test, &y_test);
    
    //call knn
    int predicted_class = knn(X_train, y_train, X_test);
    
    
    // printf("Predicted label: %d True label: %d\n", predicted_class, (int)y_test[randId]);
    
     
	free(X_train);
	free(y_train);

	free(X_test);
	free(y_test);
    
    return 0;
}
