import numpy as np
import torch.nn as nn
import torch

class AutoEncoderNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        """
        Your autoencoder model definition should go here
        """
        
        # input to hidden layer of dimension output_dim (mostly 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        
        # from hidden layer to output of dimension input_dim
        self.decoder = nn.Sequential(nn.Linear(output_dim, input_dim))
        
        
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function should map a given data matrix onto the bottleneck hidden layer
        :param x: the input data matrix of type torch.Tensor
        :return: the resulting projected data matrix of type torch.Tensor
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Your autoencoder model's forward pass operations should go here
        :param x: the input data matrix of type torch array
        :return: the neural network output as torch array
        """
        return self.decoder(self.encoder(x))

class AutoEncoder:

    def __init__(self, input_dim: int, projection_dim: int, learning_rate: float, iteration_count: int):
        """
        Initializes the Auto Encoder method
        :param input_dim: the input data space dimensionality
        :param projection_dim: the projection space dimensionality
        :param learning_rate: the learning rate for the auto encoder neural network training
        :param iteration_count: the number epoch for the neural network training
        """
        self.iteration_count = iteration_count
        """
            Your optimizer and loss definitions should go here
        """
        
        ## setting up the neural network architecture
        self.autoencoder_model = AutoEncoderNetwork(input_dim, projection_dim)
        self.optim = torch.optim.Adam(self.autoencoder_model.parameters(), lr = learning_rate)
        
        
        

    def fit(self, x: torch.Tensor) -> None:
        """
        Trains the auto encoder nn on the given data matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should train the auto encoder to minimize the reconstruction error
        please do not forget to put the neural network model into the training mode before training
        """
        
        # setting it to training mode so it updates
        self.autoencoder_model.train()
        
        #converting input to tensor (otherwise was getting error)
        x = torch.tensor(x, dtype=torch.float32)
        
        #training the model
        for i in range(self.iteration_count):
            self.optim.zero_grad()
            output = self.autoencoder_model(x)
            loss = nn.MSELoss()(output, x)
            loss.backward()
            self.optim.step()
        
    
    
        

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        After training the nn a given dataset,
        this function uses the learned model to project new data instances
        :param x: the data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        please do not forget to put the neural network model into the evaluation mode before projecting data instances
        """
        
        #setting it to evaluation mode so it doesnt update values
        self.autoencoder_model.eval()
        
        #converting input to tensor (otherwise was getting error)
        x = torch.tensor(x, dtype=torch.float32)
        
        with torch.no_grad():
            return self.autoencoder_model.project(x)
        
