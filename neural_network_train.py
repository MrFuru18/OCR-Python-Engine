import data_processing
import datetime
import time
import numpy as np


class DNN:
    def __init__(self, sizes, epochs, lr):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

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
        epo = 1
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
            self.createModel('data/weightsNN/Epoch' + str(epo) + '/')
            epo += 1


    def createModel(self, path):
        params = self.params

        listW1 = []
        for a in params["W1"]:
            values = []
            for v in a:
                values.append(v)
            listW1.append(values)
        for f in range(len(listW1)):
            output_file = path + 'W1' + 'N' + str(f) + '.txt'
            weights_to_txt(listW1[f], output_file)

        listW2 = []
        for a in params["W2"]:
            values = []
            for v in a:
                values.append(v)
            listW2.append(values)
        for f in range(len(listW2)):
            output_file = path + 'W2' + 'N' + str(f) + '.txt'
            weights_to_txt(listW2[f], output_file)

        listW3 = []
        for a in params["W3"]:
            values = []
            for v in a:
                values.append(v)
            listW3.append(values)
        for f in range(len(listW3)):
            output_file = path + 'W3' + 'N' + str(f) + '.txt'
            weights_to_txt(listW3[f], output_file)


def weights_to_txt(weights, output_file):
    data_processing.Files.list_to_txt(weights, output_file)


def main():
    train_file = open("data/mnist/mnist_train.csv", 'r')
    train_list = train_file.readlines()
    train_file.close()

    test_file = open("data/mnist/mnist_test.csv", 'r')
    test_list = test_file.readlines()
    test_file.close()


    np.random.shuffle(train_list)
    np.random.shuffle(test_list)


    dnn = DNN(sizes=[784, 256, 64, 10], epochs=60, lr=0.002)
    dnn.train(train_list, test_list, 10)



begin_time = datetime.datetime.now()
main()
print(datetime.datetime.now() - begin_time)