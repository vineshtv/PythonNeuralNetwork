""" This module contains the class definition for the neural network """
import numpy as np

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

    def train(self):
        ''' This function trains the neural network '''
        pass

    def query(self):
        ''' Query the neural network '''
        pass
