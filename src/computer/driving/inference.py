"""
Calculate prediction given a model
"""
import torch
from torchvision import transforms
import collections
from inference_model import InferenceNet
import typing
from variables import CurrentState, Command
import numpy as np

import torchvision.transforms.functional as TF

import imgaug.augmenters as iaa

class Inference():

    def __init__(self, checkpoint_path: str, num_classes: int, smoothness: int = 1):
        """Load model from checkpoint and move to gpu if possible.

        Args:
            checkpoint_path (str): Path to model
            num_classes (int): Number of classes in the model
            smoothness (int, optional): Defines the smoothness of the prediction. 
                Value between 1 and 10 where 1 means no smoothing. Defaults to 1.

        Raises:
            Exception: If smoothness is not in [1,10]
        """
        
        if not 0 < smoothness <= 10:
            raise Exception("smoothness should be between 1 and 10")

        self.NET = InferenceNet(num_classes=num_classes)
        self.NET.load_state_dict(torch.load(checkpoint_path))
        self.NET.eval()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.NET.to(self.device)

        self.translate_prediction = [Command.left, Command.right, Command.reverse, Command.forward, Command.stop]

        self.last_predictions_buffer = collections.deque(maxlen=smoothness)

    def infer(self, observation: np.ndarray, condition: int) -> str:
        """Calculates inference and returns class of prediction

        Args:
            observation (np.ndarray): Image of observation as rgb array
            condition (int): Condition

        Returns:
            int: Resulting Command of inference
        """
        observation = self._transform_observation(observation)
        observation = observation.to(self.device)

        output = self.NET(observation, condition)

        _, pred = torch.max(output, 1)

        prediction = pred.cpu().detach().numpy()
        
        return self.translate_prediction[prediction[0]]

    def _transform_observation(self, observation: np.ndarray) -> np.ndarray:
        """Resizes observation image to 224x224 and normalizes it 

        Args:
            observation (np.ndarray): Image of observation as bgr array

        Returns:
            np.ndarray: Resized and normalized rgb observation image
        """
        observation = observation[:, :, [2, 1, 0]]

        resize = iaa.Resize({"height": 224, "width": 224})
        observation = resize.augment_image(observation)

        observation = TF.to_tensor(observation)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        observation = normalize(observation)

        observation = observation.unsqueeze(0)
        return observation

    def inference_loop(self, current_state: CurrentState) -> None:
        """Endless loop to always calculate new inference

        Args:
            current_state (CurrentState): Current state variables
        """
        while True:

            current_values = current_state.get_current()
            
            observation = current_values["observation"]
            condition = current_values["condition"]

            if observation is not None and condition is not None:
                
                pred = self.infer(observation, condition)
                prediction = self._prediction_from_buffer(pred)

                current_state.set_last_predicted(observation, prediction, condition)

    def _prediction_from_buffer(self, prediction: str) -> str:
        """Smoothes prediction

        Args:
            prediction (str): Prediction

        Returns:
            str: New prediction
        """

        self.last_predictions_buffer.append(prediction)

        return max(set(self.last_predictions_buffer), key = self.last_predictions_buffer.count) 
