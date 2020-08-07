""" 
Main script to run the car.
Streaming is based on the picamera docs:
https://picamera.readthedocs.io/en/release-1.13/recipes2.html#rapid-capture-and-streaming
"""
import socket
import struct
import time
import picamera
import yaml
import threading

from simple_drive import SimpleDrive
from split_frames import SplitFrames
from variables import CurrentState


def main():
    """Main driving function
    """  

    with open("config.yml") as cf:
        config = yaml.safe_load(cf.read())

    # setup connection
    client_socket = socket.socket()
    client_socket.connect((config["ip"], 8000))
    connection = client_socket.makefile('rwb')
    
    # setup the current state
    current_state = CurrentState()

    # setup driving thread
    simple_drive = SimpleDrive(current_state)
    simple_drive_thread = threading.Thread(target=simple_drive.move)
    simple_drive_thread.setDaemon(True)
    simple_drive_thread.start()

    try:
        output = SplitFrames(client_socket,connection,current_state)
        with picamera.PiCamera(framerate=config["fps"]) as camera:
            time.sleep(2)
            start = time.time()
            camera.resolution = (config["resolutionX"], config["resolutionY"])
            
            camera.start_recording(output, format='mjpeg')

            while True:

                camera.wait_recording(1)

                if output.end_stream:
                    camera.stop_recording()
                    break
            
    finally:
        try:
            connection.close()
            client_socket.close()
        except:
            pass

        finish = time.time()
    
    print('Sent %d images in %d seconds at %.2ffps' % (
        output.count, finish-start, output.count / (finish-start)))

if __name__ == '__main__':
    main()