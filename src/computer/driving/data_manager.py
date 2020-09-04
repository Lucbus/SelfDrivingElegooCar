"""
Manages the data buffer for recording
"""
import time
import numpy as np
import cv2
from pathlib import Path

class DataManager():
    def __init__(self, data_dir: Path):
        """Inits the data manager

        Args:
            data_dir (Path): Path to data directory
        """             
        print("init datamanager")
        self.data_dir = data_dir
        self.data = []
        self.next_save = time.time()

    def save_data(self) -> None:
        """Saves the data to the hard drive
        """        
        basename = time.strftime("%Y%m%d-%H%M%S")
        
        for i, observation in enumerate (self.data):
            filename = self.data_dir / "{}_{:05d}.npy".format(basename, i) 
            np.save(filename, observation)

        self.data = []

    def reset_data(self) -> None:
        """Resets the current data
        """        
        self.data = []

    def append_data(self, image: np.ndarray, action: int, condition: int) -> None:
        """Transforms the observation image and adds the observation to the data

        Args:
            image (np.ndarray): Observation image as bgr array
            action (int): Corresponding action
            condition (int): Condition
        """        
        current_time = time.time()
        if current_time > self.next_save:
            # decode again to remove overlay text and convert to rgb
            observation = image[:, :, [2, 1, 0]]
            observation = cv2.resize(observation, (320,240))

            self.data.append({"observation": observation, "action": action, "condition": condition})
            # Make shure between the data samples are at least 0.15 sec 
            self.next_save = current_time + 0.15