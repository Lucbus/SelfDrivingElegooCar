"""
Extends the network model with a forward function to infere single images
"""
import torch
from model import Net
from variables import Command

class InferenceNet(Net):

    def forward(self, observation: torch.Tensor, condition: int) -> torch.Tensor:
        """Forward pass for inference

        Args:
            observation (torch.Tensor): Image of size 224x224x3
            condition (int): Condition

        Returns:
            torch.Tensor: Prediction
        """        

        x = self.resnet_model(observation)

        if condition == Command.left:
            x = self.fc[1](x)
        elif condition == Command.right:
            x = self.fc[2](x)
        else:
            x = self.fc[0](x)

        return x
