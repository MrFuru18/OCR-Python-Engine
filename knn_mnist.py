import data_processing
#    def write_to_txt(output, path):
#        f = open(path + "our_test.txt", 'w')
#        f.write(str(output))
#        f.close()
#for pred in y_pred:
#write_to_txt(pred, OUTPUT_DIR)

DATA_DIR = 'data/'
INPUT_DIR = 'input/'
OUTPUT_DIR = 'knn_output/'
DATASET = 'mnist'
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels.idx1-ubyte'

class KNNmnist:
    output = []
    def __init__(self):
        n=[0]


def main():
    number_of_train_images = 100
    number_of_test_images = 5
    X_train = data_processing.MNIST.load_images(TRAIN_DATA_FILENAME, number_of_train_images)
    y_train = data_processing.MNIST.load_labels(TRAIN_LABELS_FILENAME, number_of_train_images)
    X_test = data_processing.MNIST.load_images(TEST_DATA_FILENAME)
    y_test = data_processing.MNIST.load_labels(TEST_LABELS_FILENAME, number_of_test_images)
    k = 5

    print(len(X_train[0]))
    print(len(X_test[0]))

    X_train = data_processing.MNIST.to_list(X_train)
    X_test = data_processing.MNIST.to_list(X_test)

    print(len(X_train[0]))
    print(len(X_test[0]))



if __name__ == '__main__':
    main()