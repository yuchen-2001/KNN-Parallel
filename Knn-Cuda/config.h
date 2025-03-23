#define NTRAIN 1000
#define NTEST 100
#define NFEATURES 4						// Number of features (columns) in th each training example
#define NCLASSES 3						// Number of labels/ classes
#define K 11							// Hyperparameter K in KNN
#define TOPN 3							// Get the top N predictions
#define THREADS_PER_BLOCK 2048
#define X_TRAIN_PATH "../datasets/medium/X_train.csv"
#define Y_TRAIN_PATH "../datasets/medium/y_train.csv"
#define X_TEST_PATH "../datasets/medium/X_test.csv"
#define Y_TEST_PATH "../datasets/medium/y_test.csv"


// Array containing list of labels. Make changes 
char classes[NCLASSES][25] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};