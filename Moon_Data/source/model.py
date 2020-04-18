import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # define all layers, here
        self.layer_input = nn.Linear(input_dim, hidden_dim)
        
        self.layer_output = nn.Linear(hidden_dim, output_dim)
        
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)
        
        # output signal activation
        self.activation = nn.Sigmoid()
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
                
        # add hidden layer, with relu activation function
        x_output_1 = F.relu(self.layer_input(x))
        
        # add dropout layer
        x_dropped_out_1 = self.dropout(x_output_1)
        
        # add hidden layer, with relu activation function
        x_output_2 = self.layer_output(x_dropped_out_1)
        
        x_output_activated = self.activation(x_output_2)
        
        return x_output_activated