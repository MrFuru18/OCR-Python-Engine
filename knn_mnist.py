import data_processing
import datetime

TEST_DATA = 'data/mnist/t10k-images.idx3-ubyte'
TEST_LABELS = 'data/mnist/t10k-labels.idx1-ubyte'
TRAIN_DATA = 'data/mnist/train-images.idx3-ubyte'
TRAIN_LABELS = 'data/mnist/train-labels.idx1-ubyte'

'''
TEST_DATA = 'data/emnist-letters/emnist-letters-test-images-idx3-ubyte'
TEST_LABELS = 'data/emnist-letters/emnist-letters-test-labels-idx1-ubyte'
TRAIN_DATA = 'data/emnist-letters/emnist-letters-train-images-idx3-ubyte'
TRAIN_LABELS = 'data/emnist-letters/emnist-letters-train-labels-idx1-ubyte'
'''

class KNNmnist:
    @staticmethod
    def euclidean_distance(v1, v2):
        distance = 0
        for i in range(len(v1)):
            distance += (int.from_bytes(v1[i], 'big') - int.from_bytes(v2[i], 'big'))**2            #szybsze dla obrazów png
            #distance += (v1[i] - v2[i]) ** 2                                            # szybsze do testowania dokładności
        distance = distance**(1/2)
        return distance

    @staticmethod
    def minkowski_metric(v1, v2, m):
        dim = len(v1)-1
        distance = 0
        for i in range(dim):
            distance += abs(int.from_bytes(v1[i], 'big') - int.from_bytes(v2[i], 'big'))**m         #szybsze dla obrazów png
            #distance += abs(v1[i] - v2[i]) ** m                                        # szybsze do testowania dokładności
        distance = distance**(1/m)
        return distance

    @staticmethod
    def find_most_frequent(list):
        counter = 0
        num = list[0]
        for i in list:
            curr_frequency = list.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i
        return num

    @staticmethod
    def clustering(X_train, y_train, sample, k):
        distances = []
        for i in range(len(X_train)):
            #distances.append(KNNmnist.euclidean_distance(sample, X_train[i]))
            distances.append(KNNmnist.minkowski_metric(sample, X_train[i], k))

        sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])

        winners_idx = []
        winners = []
        for i in range(k):
            winners_idx.append(sorted_distances[i][0])
            winners.append(y_train[winners_idx[i]])

        print(winners)
        winner = KNNmnist.find_most_frequent(winners)

        return winner



def test_accuracy(X_train, y_train, X_test, y_test, k):
    correct = 0
    for i in (range(len(X_test))):
        if KNNmnist.clustering(X_train, y_train, X_test[i], k) == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test) * 100
    return accuracy

def test_k(X_train, y_train, X_test, y_test, output_dir):
    for k in range(1,9):
        output_file = output_dir + 'Accuracy k=' + str(k) + '.txt'
        data_processing.Files.write_to_txt(test_accuracy(X_train, y_train, X_test, y_test, k), output_file)

def get_list_of_indexes(y):
    list_of_digits_indexes = []
    for digit in range(10):
        digit_indexes = []
        for index in range(len(y)):
            if digit == y[index]:
                digit_indexes.append(index)
        list_of_digits_indexes.append(digit_indexes)
    return list_of_digits_indexes

def test_accuracy_for_digit(X_train, y_train, X_test, y_test_list_of_indexes, digit, k):
    correct = 0
    for index in y_test_list_of_indexes[digit]:
        if KNNmnist.clustering(X_train, y_train, X_test[index], k) == digit:
            correct += 1
    accuracy = correct / len(y_test_list_of_indexes[digit]) * 100
    return accuracy

def analyze_input_set(X_train, y_train, X_test, k):
    y_output = []
    for sample in X_test:
        y_output.append(KNNmnist.clustering(X_train, y_train, sample, k))
    return y_output

def main():
    number_of_train_images = 6000
    number_of_test_images = 1000
    k = 5

    X_train = data_processing.MNIST.load_images(TRAIN_DATA, number_of_train_images)
    y_train = data_processing.MNIST.load_labels(TRAIN_LABELS, number_of_train_images)
    X_train = data_processing.Image.dataset_to_list(X_train)

    '''
    X_test = data_processing.MNIST.load_images(TEST_DATA, number_of_test_images)
    y_test = data_processing.MNIST.load_labels(TEST_LABELS, number_of_test_images)
    X_test = data_processing.Image.dataset_to_list(X_test)
    
    test_k(X_train, y_train, X_test, y_test, 'knn_output/accuracy_euclidean/')
    test_k(X_train, y_train, X_test, y_test, 'knn_output/accuracy_minkowski/')
    ''''''
    y_test_list_of_indexes = get_list_of_indexes(y_test)
    
    for i in range(10):
        output_file = 'knn_output/accuracy_minkowski/Accuracy k=5 for ' + str(i) + '.txt'
        data_processing.Files.write_to_txt(test_accuracy_for_digit(X_train, y_train, X_test, y_test_list_of_indexes, i, k),
                                           output_file)
    '''

    images = data_processing.Files.read_images('input/')
    images = data_processing.Image.dataset_to_list(images)

    data_processing.Files.input_to_txt(analyze_input_set(X_train, y_train, images, k), 'knn_output/', 'input/')




begin_time = datetime.datetime.now()
main()
print(datetime.datetime.now() - begin_time)
