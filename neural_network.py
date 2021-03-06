""" This module contains the class definition for the neural network """
import numpy as np
import scipy.special

class NeuralNetwork(object):
    ''' NeuralNetwork class definition '''
    # Init
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Learning rate
        self.lrate = learningrate

        # link weights wih and who
        # random weights are sampled from a normal distribution centred around 0.0
        # and a standard deviation of 1/sqrt(number of incoming links)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # activation function here is the sigmoid function 
        # expit is the inbuilt sigmoid function in scipy
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        ''' This function trains the neural network '''
        # Convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate the signals into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)
      
        # Calculate the errors 
        # output error is target - output
        output_errors = targets - final_outputs

        # hidden layer error is output_errors split by the weights.
        hidden_errors = np.dot(self.who.T, output_errors)

        #update the weights between hidden and output layer
        self.who += self.lrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        #update the weights between the input and the hidden layer
        self.wih += self.lrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        ''' Query the neural network '''
        # convert inputs list into 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate the signals into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
