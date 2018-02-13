#!/usr/bin/env python

from sklearn.datasets import fetch_mldata
from neural_network import NeuralNetwork 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def main():
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    # Learning rate = 0.3
    learning_rate = 0.3

    # Create an instance of neural network
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]

    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
   
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    
    # test the neural network

    # scorecard for how well the network performs
    scorecard = []

    index = 0;
    for record in X_test:
        scaled_input = (np.asfarray(record) / 255.0 * 0.99) + 0.01
        correct_label = y_test[index]
        index = index + 1
        outputs = nn.query(scaled_input)
        label = np.argmax(outputs)

        #append correct or incorrect to list
        if(label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)

    #calulate the performance
    scorecard_array = np.asarray(scorecard)
    print("sum = ", scorecard_array.sum())
    print("Performance = ", scorecard_array.sum() / scorecard_array.size)

if __name__ == "__main__":
    main()
