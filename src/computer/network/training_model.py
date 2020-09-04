"""
Definition of forward function for training model
"""
import torch
from model import Net

class TrainingNet(Net):

    def forward(self, observation: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of observation and condition.

        Args:
            observation (torch.Tensor): Batch of Observations
            condition (torch.Tensor): Batch of conditions

        Returns:
            torch.Tensor: Batch of results
        """        
        x = self.resnet_model(observation)

        result = 0
        result += self.fc[0](x) * (condition == 1).float().unsqueeze_(-1)
        result += self.fc[1](x) * (condition == 2).float().unsqueeze_(-1)
        result += self.fc[2](x) * (condition == 4).float().unsqueeze_(-1)

        return result