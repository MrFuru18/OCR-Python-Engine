import data_processing
import datetime
import time
import numpy as np

class DNN:
    def __init__(self, sizes, epochs, lr, lw):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr
        self.lw = lw

        # number of nodes in each layer
        input_layer = self.sizes[0]
        hidden_1 = self.sizes[1]
        hidden_2 = self.sizes[2]
        output_layer = self.sizes[3]

        self.params = {
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }


    def replace_weights_values(self):
        for i in range(128):
            for j in range(728):
                self.params["W1"][i][j] = self.lw[0][i][j];

        for i in range(64):
            for j in range(128):
                self.params["W2"][i][j] = self.lw[0][i][j];

        for i in range(10):
            for j in range(64):
                self.params["W3"][i][j] = self.lw[0][i][j];

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def forward_pass(self, x_train):
        params = self.params

        # input layer activations becomes sample
        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):
        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):

        for key, value in changes_to_w.items():
            self.params[key] -= self.lr * value

    def compute_accuracy(self, test_data, output_nodes):
        predictions = []

        for x in test_data:
            all_values = x.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            output = self.forward_pass(inputs)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(targets))

        return np.mean(predictions)

    def train(self, train_list, test_list, output_nodes):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x in train_list:
                all_values = x.split(',')
                # scale and shift the inputs
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = np.zeros(output_nodes) + 0.01
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                output = self.forward_pass(inputs)
                changes_to_w = self.backward_pass(targets, output)
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(test_list, output_nodes)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration + 1, time.time() - start_time, accuracy * 100
            ))



    def classify(self, x):
        _x = []
        for p in x:
            p = (p / 255.0 * 0.99) + 0.01
            _x.append(p)
        output = self.forward_pass(_x)
        pred = np.argmax(output)
        return pred




def weights_to_txt(weights, output_file):
    data_processing.Files.list_to_txt(weights, output_file)

def get_list_of_weights(path):
    list_of_weights = []
    list_of_weightsN1 = []
    for N1 in range(256):
        file_path = path + 'W1N' + str(N1) + '.txt'
        list_of_weightsN1.append(data_processing.Files.read_list(file_path))
    list_of_weightsN2 = []
    for N2 in range(64):
        file_path = path + 'W2N' + str(N2) + '.txt'
        list_of_weightsN2.append(data_processing.Files.read_list(file_path))
    list_of_weightsN3 = []
    for N3 in range(10):
        file_path = path + 'W3N' + str(N3) + '.txt'
        list_of_weightsN3.append(data_processing.Files.read_list(file_path))

    list_of_weights.append(list_of_weightsN1)
    list_of_weights.append(list_of_weightsN2)
    list_of_weights.append(list_of_weightsN3)

    return list_of_weights

def analyze_input_set(dnn, X_test):
    y_output = []
    for sample in X_test:
        y_output.append(dnn.classify(sample))
    return y_output



def main():
    list_of_weights = get_list_of_weights('data/weightsNN/Epoch8/')

    images = data_processing.Files.read_images('input/')
    images = data_processing.Image.dataset_to_list(images)


    dnn = DNN(sizes=[784, 256, 64, 10], epochs=1, lr=0.002, lw=list_of_weights)
    dnn.replace_weights_values()
    data_processing.Files.input_to_txt(analyze_input_set(dnn, images), 'nn_output/', 'input/')



begin_time = datetime.datetime.now()
main()
print(datetime.datetime.now() - begin_time)
