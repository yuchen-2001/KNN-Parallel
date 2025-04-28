#define NTRAIN 50000
#define NTEST 2000
#define NFEATURES 4						// Number of features (columns) in th each training example
#define NCLASSES 3						// Number of labels/ classes
#define K 11							// Hyperparameter K in KNN
#define TOPN 3							// Get the top N predictions
#define THREADS_PER_BLOCK 512
#define X_TRAIN_PATH "../../../datasets/large/X_train.csv"
#define Y_TRAIN_PATH "../../../datasets/large/y_train.csv"
#define X_TEST_PATH "../../../datasets/large/X_test.csv"
#define Y_TEST_PATH "../../../datasets/large/y_test.csv"


// Array containing list of labels. Make changes 
char classes[NCLASSES][25] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};