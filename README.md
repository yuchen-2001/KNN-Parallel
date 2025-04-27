# Test Machine
- Host: `ece017.ece.local.cmu.edu`

# How to use
## To compile the .cu file
1. **Move into the scripts folder**
    - `cd ./Knn-Cuda/scripts`
2. **Compile .cu files by running shell scripts**
    - Small Size Dataset: `./run_knn_test_small.sh`
    - Medium Size Dataset: `./run_knn_test_medium.sh`
    - Large Size Dataset: `./run_knn_test_large.sh`

## To run the .out file
1. **Move into the outputs folder**
    1.  `cd ../outputs` (`./Knn-Cuda/outputs/`)
    2.  `cd small` / `cd medium` / `cd large`
> [!important]
> You must ensure you enter the small/medium/large folder. Otherwise, there will be path errors.

2. **Run .out files**: `./knnInCuda_xxx.out`

> [!NOTE]
> `xxx` should be replaced to the shape of block. Use `ls` to check the avaliable shapes in each folders.
