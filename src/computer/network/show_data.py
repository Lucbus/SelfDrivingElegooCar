"""
Script to view the recorded data
"""
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class ShowData():
    def __init__(self, data_dir: Path):
        """Initializes ShowData

        Args:
            data_dir (Path): Path to data
        """        
        self.data_dir = data_dir
        self.class_names = ['w', 'a', 's', 'd', 'x', 'e', 'q']

    def show(self):
        """First shows a bar plot of the action distribution. Then all observations are shown.
            This function is for demonstration purposes only. Use left and right arrow keys to navigate
        """        

        data = []
        actions = []
        conditions = []
        for i in self.data_dir.rglob('*.npy'):
            obs = np.load(i, allow_pickle=True).item()

            new_file = {
                "obs": obs['observation'],
                "condition": obs['condition'],
                "action": self.class_names[obs['action']-1],
                "name": i.name
            }

            data.append(new_file)

            actions.append(self.class_names[obs['action']-1])
            conditions.append(obs['condition'])

        unique_actions, count_actions = np.unique(actions, return_counts=True)
        
        fig, ax = plt.subplots()
        plt.bar(unique_actions, count_actions)
        plt.show()

        #show the individual observations:
        num_files = len(data)
        i = 0

        while i < num_files:
            print("{}/{}".format(i+1, num_files))
            
            obs = data[i]

            image = cv2.cvtColor(obs['obs'], cv2.COLOR_BGR2RGB)

            cv2.putText(image, obs["action"], (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            cv2.putText(image, obs["name"], (10,220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

            if obs["condition"] == 1:
                arrow_start = (300,30)
                arrow_end = (300,10)
            elif obs["condition"] == 2:
                arrow_start = (310,20)
                arrow_end = (290,20)
            elif obs["condition"] == 3:
                arrow_start = (300,10)
                arrow_end = (300,30)
            elif obs["condition"] == 4:
                arrow_start = (290,20)
                arrow_end = (310,20)

            cv2.arrowedLine(image, arrow_start, arrow_end, (0,0,255), 2, tipLength = 0.5)

            cv2.imshow('image', image)

            cvkey = cv2.waitKeyEx(0)

            if cvkey == 27: #escape
                break
            elif cvkey == 2555904: #right
                i += 1
            elif cvkey == 2424832 and i > 0: #left
                i -= 1
