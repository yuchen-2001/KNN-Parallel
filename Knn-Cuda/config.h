#define NTRAIN 10000
#define NTEST 500
#define NFEATURES 4						// Number of features (columns) in th each training example
#define NCLASSES 3						// Number of labels/ classes
#define K 11							// Hyperparameter K in KNN
#define TOPN 3							// Get the top N predictions
// Drop THREADS_PER_BLOCK from 2048 to 1024 because Tesla T4 only support 1024
#define THREADS_PER_BLOCK 32
// Define blockDim.x and blockDim.y
#define BLOCK_X 16
#define BLOCK_Y 32
#define X_TRAIN_PATH "../../datasets/medium/X_train.csv"
#define Y_TRAIN_PATH "../../datasets/medium/y_train.csv"
#define X_TEST_PATH "../../datasets/medium/X_test.csv"
#define Y_TEST_PATH "../../datasets/medium/y_test.csv"


// Array containing list of labels. Make changes 
// char classes[NCLASSES][25] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};