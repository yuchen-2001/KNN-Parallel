#include "utils.h"
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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