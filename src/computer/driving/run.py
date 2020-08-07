"""
Main script to run the program
"""

import cv2
import time
import os
import numpy as np
import threading
import argparse
import yaml
from pathlib import Path
import time

from inference import Inference
from variables import Command, CurrentState, Mode
from network import Network
from gui import Gui
from data_manager import DataManager
from manager import Manager

def main():
    """Inits all the variables, calls the config and starts the main loop. 
        After exiting the main loop this ends the connection.
    """    
    parser = argparse.ArgumentParser(description="Path to checkpoint")
    parser.add_argument('--checkpoint', help='File name of the trained weights.',
                        default='')

    args = parser.parse_args()
    checkpoint_name = args.checkpoint

    
    with open("config.yml") as cf:
        config = yaml.safe_load(cf.read())

    project_path = Path.cwd().parent.parent.parent
    path = project_path / config['data_path']
    data_manager = DataManager(path)


    connection = Network(config['ip'], config['port'])


    application_state = CurrentState()

    if checkpoint_name == "":
        application_mode = Mode.manual
    else:
        application_mode = Mode.inference
        # setup inference thread
        checkpoint_path = project_path / config['checkpoint_path'] / checkpoint_name
        inf = Inference(checkpoint_path, 5)
        
        inference_thread = threading.Thread(target=inf.inference_loop, args=((application_state,)))
        inference_thread.setDaemon(True)
        inference_thread.start()

    program_manager = Manager(application_mode, connection, application_state, data_manager)

    program_manager.main_loop()

    # quit
    connection.send_command(Command.end_stream)
    time.sleep(.1)
    connection.close()

if __name__ == '__main__':
    main()
