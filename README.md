# Problem Statement
K-Nearest Neighbors (KNN) is a simple yet widely used classifier whose basic implementation requires a full distance computation between every query and every training point, making it slow for large datasets. We present a high-performance GPU solution that re-architects KNN for NVIDIA CUDA. Our design uses a two-dimensional tiling strategy with shared-memory caching to minimize global-memory traffic during distance evaluation, and a heap-based selection kernel to extract only the K smallest distances without sorting the entire array. Batched execution with overlapping data transfers via CUDA streams enables datasets that exceed on-device memory. Compared with a public baseline GPU implementation, our approach achieves better speed-up on medium-scale and large-scale benchmarks while retaining the accuracy metrics. The resulting system demonstrates that careful algorithmic reformulation combined with GPU-aware optimization (coalesced access, shared memory, warp-level cooperation) can transform KNN from a quadratic-time bottleneck into an interactive-speed primitive suitable for real-world, large-scale machine-learning pipelines.


# Documentation
- We have drafted a slideshow [(Click here)](https://docs.google.com/presentation/d/1vzvIeqpRzkzitCd1WYA3cXElE76XZvB7/edit?usp=sharing&ouid=107406395363399456335&rtpof=true&sd=true) and a report [(Click here)](https://drive.google.com/file/d/1PEYKyO7yjyX8hCyDGyY4dC7xo4ALtCch/view?usp=sharing)  about the project.


# Test Machine
- Host: `ece017.ece.local.cmu.edu`


# How to use

### To compile the .cu file
1. **Move into the scripts folder**
    - `cd ./Knn-Cuda/scripts`
2. **Compile .cu files by running shell scripts**
    - Small Size Dataset: `./run_knn_test_small.sh`
    - Medium Size Dataset: `./run_knn_test_medium.sh`
    - Large Size Dataset: `./run_knn_test_large.sh`

### To run the .out file
1. **Move into the outputs folder**
    1.  `cd ../outputs` (`./Knn-Cuda/outputs/`)
    2.  `cd small` / `cd medium` / `cd large`
> [!important]
> You must ensure you enter the small/medium/large folder. Otherwise, there will be path errors.

2. **Run .out files**: `./knnInCuda_xxx.out`

> [!NOTE]
> `xxx` should be replaced to the shape of block. Use `ls` to check the avaliable shapes in each folders.
