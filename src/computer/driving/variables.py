"""
Global variables to share between inference and main thread. 
"""
from enum import IntEnum, Enum
import threading
import typing
import numpy as np

class Command(IntEnum):
    """Maps a command to an int value
    """    
    forward = 1
    left = 2
    reverse = 3
    right = 4
    stop = 5
    turn_left = 6
    turn_right = 7
    end_stream = 8
    no_command = -1

class Prediction(Enum):
    """Maps a command to a string to show in gui
    """    
    forward = "w"
    left = "a"
    reverse = "s"
    right = "d"
    stop = "x"
    no_command = ""

class Mode(IntEnum):
    """Mode in which the application can run
    """
    manual = 1
    inference = 2

class GUIColor(Enum):
    """Predefined colors for gui
    """
    red = (0,0,255)
    green = (0,255,0)

class CurrentState():
    """Class to hold the global application state
    """    
    def __init__(self) -> None:
        """Inits the state. The initial condition is Command.forward, the initial 
            predicted action is Command.no_command
        """        
        self.last_predicted_observation = None
        self.last_predicted_action = Command.no_command
        self.last_predicted_condition = Command.forward
        self.current_condition = Command.forward 
        self.current_observation = None
        
        self.current_lock = threading.Lock()
        self.last_lock = threading.Lock()

    def get_last_predicted(self) -> dict:
        """Returns last prediction

        Returns:
            dict: Containing "observation","action" and "condition"
        """        
        with self.last_lock:
            return {"observation": self.last_predicted_observation,
                    "action": self.last_predicted_action,
                    "condition": self.last_predicted_condition}

    def set_last_predicted(self, observation: np.ndarray, action: Command, condition: Command) -> None:
        """Setter for last prediction

        Args:
            observation (np.ndarray): Image of last prediction
            action (Command): Action of last prediction
            condition (Command): Condition of last prediction
        """        
        with self.last_lock:
            self.last_predicted_observation = observation
            self.last_predicted_action = action
            self.last_predicted_condition = condition

    def get_current(self) -> dict:
        """Returns the current Observation and Condition

        Returns:
            dict: "condition"(Command) and "observation"(np.ndarray) of current state
        """        
        with self.current_lock:
            return {"condition": self.current_condition,
                    "observation": self.current_observation}

    def set_current(self, observation: np.array, condition: Command) -> None:
        """Setter for current state

        Args:
            observation (np.array): Image of current observation
            condition (Command): Current condition
        """        
        with self.current_lock:
            self.current_condition = condition
            self.current_observation = observation
