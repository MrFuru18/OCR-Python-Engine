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


class WeightsModel:
    @staticmethod
    def normalize_weights(list_of_weights):
        for digit in list_of_weights:
            minElement = min(digit)
            maxElement = max(digit)
            for pixel in range(len(digit)):
                digit[pixel] = (digit[pixel] - minElement) / (maxElement - minElement)
        return list_of_weights

    @staticmethod
    def set_weights(X_train, list_of_indexes):
        list_of_weights = []
        for digit in list_of_indexes:
            digit_weights = []
            for pixel in range(len(X_train[0])):
                sum_of_pixel_values = 0
                for index in digit:
                    sum_of_pixel_values += X_train[index][pixel]
                weight_of_pixel = sum_of_pixel_values / len(digit)
                digit_weights.append(weight_of_pixel)
            list_of_weights.append(digit_weights)
        return WeightsModel.normalize_weights(list_of_weights)

    @staticmethod
    def classify(list_of_weights, sample):
        max_element = 0
        for digit in range(len(list_of_weights)):
            score = 0
            for pixel_value in range(len(sample)):
                score += float(list_of_weights[digit][pixel_value]) * sample[pixel_value]
            if (score > max_element):
                max_element = score
                result = digit
        return result



def get_list_of_indexes(y):
    list_of_digits_indexes = []
    for digit in range(10):
        digit_indexes = []
        for index in range(len(y)):
            if digit == y[index]:
                digit_indexes.append(index)
        list_of_digits_indexes.append(digit_indexes)
    return list_of_digits_indexes

def weights_to_txt(weights, output_dir):
    for digit in range(len(weights)):
        output_file = output_dir + 'Weight' + str(digit) + '.txt'
        data_processing.Files.list_to_txt(weights[digit], output_file)

def create_model(X_train, y_train, path):
    list_of_indexes = get_list_of_indexes(y_train)
    weights = WeightsModel.set_weights(X_train, list_of_indexes)
    weights_to_txt(weights, path)

def get_list_of_weights(path):
    list_of_weights = []
    for digit in range(10):
        file_path = path + 'Weight' + str(digit) + '.txt'
        list_of_weights.append(data_processing.Files.read_list(file_path))
    return list_of_weights

def test_accuracy(list_of_weights, X_test, y_test):
    correct = 0
    for i in (range(len(X_test))):
        if WeightsModel.classify(list_of_weights, X_test[i]) == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test) * 100
    return accuracy

def test_accuracy_for_digit(list_of_weights, X_test, y_test_list_of_indexes, digit):
    correct = 0
    for index in y_test_list_of_indexes[digit]:
        if WeightsModel.classify(list_of_weights, X_test[index]) == digit:
            correct += 1
    accuracy = correct / len(y_test_list_of_indexes[digit]) * 100
    return accuracy

def analyze_input_set(list_of_weights, X_test):
    y_output = []
    for sample in X_test:
        y_output.append(WeightsModel.classify(list_of_weights, sample))
    return y_output

def main():
    '''
    number_of_train_images = 60000
    X_train = data_processing.MNIST.load_images(TRAIN_DATA, number_of_train_images)
    y_train = data_processing.MNIST.load_labels(TRAIN_LABELS, number_of_train_images)
    X_train = data_processing.Image.dataset_to_list(X_train)

    create_model(X_train, y_train, 'data/weights/')
    ''''''
    number_of_test_images = 10000
    X_test = data_processing.MNIST.load_images(TEST_DATA, number_of_test_images)
    y_test = data_processing.MNIST.load_labels(TEST_LABELS, number_of_test_images)
    X_test = data_processing.Image.dataset_to_list(X_test)

    list_of_weights = get_list_of_weights('data/weights/')
    data_processing.Files.write_to_txt(test_accuracy(list_of_weights, X_test, y_test), 'weights_model_output/accuracy/Accuracy.txt')

    y_test_list_of_indexes = get_list_of_indexes(y_test)

    for i in range(10):
        output_file = 'weights_model_output/accuracy/Accuracy for ' + str(i) + '.txt'
        data_processing.Files.write_to_txt(test_accuracy_for_digit(list_of_weights, X_test, y_test_list_of_indexes, i), output_file)
    '''

    #azanliza folderu input
    list_of_weights = get_list_of_weights('data/weights/')
    images = data_processing.Files.read_images('input/')
    images = data_processing.Image.dataset_to_list(images)

    data_processing.Files.input_to_txt(analyze_input_set(list_of_weights, images), 'weights_model_output/', 'input/')



begin_time = datetime.datetime.now()
main()
print(datetime.datetime.now() - begin_time)