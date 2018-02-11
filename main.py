from neural_network import NeuralNetwork 

def main():
    # number of input, hidden and output nodes
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    # Learning rate = 0.3
    learning_rate = 0.3

    # Create an instance of neural network
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    inputs = [1.0, 0.5, -1.5]
    
    # query the neural network
    output = nn.query(inputs)

    print ("output is : ", output)

if __name__ == "__main__":
    main()
