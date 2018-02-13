#!/usr/bin/env python

from __future__ import division
from sklearn.datasets import fetch_mldata
from neural_network import NeuralNetwork 
import numpy as np

def main():
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # Learning rate = 0.1
    learning_rate = 0.1

    # Create an instance of neural network
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Fetch the MNIST data
    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]

    # Split the data into test and train set
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
   
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    

    # Train the neural network
    # epochs is the number of times the training set is used to train the neural network
    epochs = 5
    for e in range(epochs):
        msg = "Training epoch " + str(e + 1) + " of " + str(epochs) + "..."
        print (msg)
        index = 0
        # Go through all the records in the training set
        for record in X_train:
            scaled_input = (np.asfarray(record) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(y_train[index])] = 0.99
            nn.train(scaled_input, targets)
            index = index + 1
            
    print("Training complete!!!")
    # test the neural network

    # scorecard for how well the network performs
    scorecard = []

    print("Testing the Neural Network.") 
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
            #print("NN prediction = ", label)
            #print("Actual label = ", correct_label)
            scorecard.append(0)

    #calulate the performance
    scorecard_array = np.asarray(scorecard)
    #print("sum = ", scorecard_array.sum())
    #print("size = ", scorecard_array.size)
    print("Performance of the Neural Network = ", scorecard_array.sum() / scorecard_array.size)

if __name__ == "__main__":
    main()
