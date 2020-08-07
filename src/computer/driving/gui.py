"""
Class to handle the gui for driving the car
"""
import numpy as np
import cv2
from variables import Command, Prediction, GUIColor

class Gui():
    def __init__(self):
        pass

    def show_state(self, image: np.ndarray, condition: Command, prediction: Command, 
                    recording: bool = False, self_driving : bool = False) -> None:
        """Opens a window to show the information to the user

        Args:
            image (np.ndarray): Current observation to show
            condition (Command): Current condition
            prediction (Command): Current prediction
            recording (bool, optional): Current recording state. 
                If currently recording this should be true. Defaults to False.
            self_driving (bool, optional): True if in self driving mode. Defaults to False.
        """        

        arrow = self._get_arrow(condition, image.shape[1])
        cv2.arrowedLine(image, arrow["start"], arrow["end"], GUIColor.red.value, 2, tipLength = 0.5)

        if self_driving:
            prediction_color = GUIColor.green.value
        else:
            prediction_color = GUIColor.red.value

        pred = Prediction[prediction.name].value

        cv2.putText(image, pred, (30,30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=prediction_color, thickness=2)

        if recording:
            cv2.putText(image, 'rec', (image.shape[1]-60,image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, GUIColor.red.value)

        if image is not None:
            cv2.imshow('stream', image)

        cv2.waitKey(1)

    def transform_image(self, data: np.ndarray) -> np.ndarray:
        """Transforms incoming one dimensional data array into a BGR image

        Args:
            data (np.ndarray): One dimensional data of image

        Returns:
            np.ndarray: BGR image
        """      
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image

    def _get_arrow(self, condition: Command, stream_width: int = 640) -> dict:
        """Returns start and end position of an arrow according to condition

        Args:
            condition (Command): Command to transform into arrow
            stream_width (int, optional): Streaming width. Defaults to 640.

        Returns:
            dict: Containing "start" and "end" of the corresponding arrow
        """        
        if condition == Command.left:
            return {"start": (stream_width - 20,20), "end": (stream_width - 40,20)}
        elif condition == Command.right:
            return {"start": (stream_width - 40,20), "end": (stream_width - 20,20)}
        else:
            return {"start": (stream_width - 30,30), "end": (stream_width - 30,10)}
