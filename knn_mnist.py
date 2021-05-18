import data_processing
import datetime

DATA_DIR = 'data/'
INPUT_DIR = 'input/'
OUTPUT_DIR = 'knn_output/'
DATASET = 'mnist'
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels.idx1-ubyte'


class KNNmnist:
    @staticmethod
    def euclidean_distance(v1, v2):
        distance = 0
        for i in range(len(v1)):
            distance += (int.from_bytes(v1[i], 'big') - int.from_bytes(v2[i], 'big'))**2
        distance = distance**(1/2)
        return distance

    @staticmethod
    def minkowski_metric(v1, v2, m):
        dim = len(v1)-1
        distance = 0
        for i in range(dim):
            distance += abs(int.from_bytes(v1[i], 'big') - int.from_bytes(v2[i], 'big'))**m
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
            distances.append(KNNmnist.euclidean_distance(sample, X_train[i]))
            #distances.append(KNNmnist.minkowski_metric(sample, X_train[i], k))

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

def analyze_input_set(X_train, y_train, X_test, k):
    y_output = []
    for sample in X_test:
        y_output.append(KNNmnist.clustering(X_train, y_train, sample, k))
    return y_output



def main():
    number_of_train_images = 2500
    number_of_test_images = 4
    k = 9

    X_train = data_processing.MNIST.load_images(TRAIN_DATA_FILENAME, number_of_train_images)
    y_train = data_processing.MNIST.load_labels(TRAIN_LABELS_FILENAME, number_of_train_images)
    X_test = data_processing.MNIST.load_images(TEST_DATA_FILENAME, number_of_test_images)
    y_test = data_processing.MNIST.load_labels(TEST_LABELS_FILENAME, number_of_test_images)

    X_train = data_processing.Image.dataset_to_list(X_train)
    X_test = data_processing.Image.dataset_to_list(X_test)

    images = data_processing.Files.read_images(INPUT_DIR)
    images = data_processing.Image.dataset_to_list(images)

    data_processing.Files.write_to_txt(analyze_input_set(X_train, y_train, images, k), OUTPUT_DIR, INPUT_DIR)

    #print(KNNmnist.clustering(X_train, y_train, X_test[0], k))
    #print(test_accuracy(X_train, y_train, X_test, y_test, k))


if __name__ == '__main__':
    begin_time = datetime.datetime.now()

    main()

    print(datetime.datetime.now() - begin_time)
