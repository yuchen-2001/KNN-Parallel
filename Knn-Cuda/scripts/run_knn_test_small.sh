#!/bin/bash
set -euo pipefail

SOURCE_FILE="../knnInCuda.cu"
CONFIG_FILE="../config.h"
UTILS_FILE="../utils.cu"
X_TRAIN_PATH="\"../../datasets/small/X_train.csv\""
Y_TRAIN_PATH="\"../../datasets/small/y_train.csv\""
X_TEST_PATH="\"../../datasets/small/X_test.csv\""
Y_TEST_PATH="\"../../datasets/small/y_test.csv\""
NTRAIN="135"
NTEST="15"

# List of block shape to test
BLOCK_X=(8 8 16 16 32)
BLOCK_Y=(4 8 8 16 16)

echo "Running tests"
echo "==============================="

# Modify config
sed -i "s|^#define X_TRAIN_PATH .*|#define X_TRAIN_PATH $X_TRAIN_PATH|" "$CONFIG_FILE"
sed -i "s|^#define Y_TRAIN_PATH .*|#define Y_TRAIN_PATH $Y_TRAIN_PATH|" "$CONFIG_FILE"
sed -i "s|^#define X_TEST_PATH .*|#define X_TEST_PATH $X_TEST_PATH|" "$CONFIG_FILE"
sed -i "s|^#define Y_TEST_PATH .*|#define Y_TEST_PATH $Y_TEST_PATH|" "$CONFIG_FILE"
sed -i "s/^#define NTRAIN .*/#define NTRAIN $NTRAIN/" "$CONFIG_FILE"
sed -i "s/^#define NTEST .*/#define NTEST $NTEST/" "$CONFIG_FILE"

for ((i = 0; i < ${#BLOCK_X[@]}; i++)); do
    bx=${BLOCK_X[$i]}
    by=${BLOCK_Y[$i]}
    
    echo
    echo "[Testing with BLOCK_X=$bx, BLOCK_Y=$by]..."

    sed -i "s/^#define BLOCK_X .*/#define BLOCK_X $bx/" "$CONFIG_FILE"
    sed -i "s/^#define BLOCK_Y .*/#define BLOCK_Y $by/" "$CONFIG_FILE"

    OUTPUT="../outputs/small/knnInCuda_${bx}x${by}.out"
    nvcc -o "$OUTPUT" "$SOURCE_FILE" "$UTILS_FILE"
done

echo
echo "All tests complete."
