#include <stdio.h>

void checkFile(FILE *f);
float *getFloatMat(int m, int n);
float *initFeatures(const char path[]);
float getMax(float *x, int n);
float *initLabels(const char path[]);