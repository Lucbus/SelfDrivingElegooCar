"""
Script to measure distance to an obstacle. The script is based on
https://tutorials-raspberrypi.de/entfernung-messen-mit-ultraschallsensor-hc-sr04/
"""    
import RPi.GPIO as GPIO
import time
import datetime
from variables import CurrentState


class Distance():
    
    def __init__(self, currentState: CurrentState):
        """
        Initializes the io pins for the distance measuring as well as the current state

        Args:
            currentState (CurrentState): Current state
        """        
        self.TRIG = 23
        self.ECHO = 24

        self.currentState = currentState

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)
        GPIO.output(self.TRIG, False)

    def measure_distance(self) -> None:
        """
        Measures the distance to obstacles in front of the car and saves
        them to the current state. 
        If no echo is detected a distance of 999 is saved.
        """
        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)

        echo_timeout = time.time() + 0.06 # > 999 cm
        timeout = False
        start = time.time()
        stop = time.time()

        while GPIO.input(self.ECHO) == 0:
            start = time.time()
            if time.time() > echo_timeout:
                timeout = True
                break

        while GPIO.input(self.ECHO) == 1:
            stop = time.time()
            if time.time() > echo_timeout:
                timeout = True
                break

        if timeout:
            self.currentState.distance = 999
        else: 
            elapsed = stop - start
            distance = elapsed * 17150 #speed of sound: 34300cm/sec

            self.currentState.distance = distance
