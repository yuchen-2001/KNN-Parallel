#!/bin/bash
set -euo pipefail

SOURCE_FILE="../knnInCuda_old.cu"
CONFIG_FILE="../config.h"
UTILS_FILE="../utils.cu"
X_TRAIN_PATH="\"../../datasets/large/X_train.csv\""
Y_TRAIN_PATH="\"../../datasets/large/y_train.csv\""
X_TEST_PATH="\"../../datasets/large/X_test.csv\""
Y_TEST_PATH="\"../../datasets/large/y_test.csv\""
NTRAIN="135"
NTEST="15"

# List of kernel counts to test
KERNELS=(32 64 128 256 512)

echo "Running tests"
echo "==============================="

# Modify config
sed -i "s|^#define X_TRAIN_PATH .*|#define X_TRAIN_PATH $X_TRAIN_PATH|" "$CONFIG_FILE"
sed -i "s|^#define Y_TRAIN_PATH .*|#define Y_TRAIN_PATH $Y_TRAIN_PATH|" "$CONFIG_FILE"
sed -i "s|^#define X_TEST_PATH .*|#define X_TEST_PATH $X_TEST_PATH|" "$CONFIG_FILE"
sed -i "s|^#define Y_TEST_PATH .*|#define Y_TEST_PATH $Y_TEST_PATH|" "$CONFIG_FILE"
sed -i "s/^#define NTRAIN .*/#define NTRAIN $NTRAIN/" "$CONFIG_FILE"
sed -i "s/^#define NTEST .*/#define NTEST $NTEST/" "$CONFIG_FILE"

for K in "${KERNELS[@]}"; do
    echo
    echo "[Testing with $K kernels]..."
    sed -i "s/^#define THREADS_PER_BLOCK .*/#define THREADS_PER_BLOCK $K/" "$CONFIG_FILE"
    OUTPUT="../outputs/small/knnInCuda_$K.out"
    nvcc -o "$OUTPUT" "$SOURCE_FILE" "$UTILS_FILE"
done

echo
echo "All tests complete."
