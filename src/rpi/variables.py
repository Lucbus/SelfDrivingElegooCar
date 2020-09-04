"""
Contains all global variables
"""
import yaml
from enum import IntEnum
import time

class CurrentState():
    def __init__(self):
        """Init variables
        """        
        with open("config.yml") as cf:
            config = yaml.safe_load(cf.read())

        self.distance = 0
        self.stop_timer = 0
        self.speed = config["speed"]
        self.direction = 0
        self.driving_time = config["drivingTime"]
        self.driving_time_turning = config["drivingTimeTurning"]
        
        self.distance_stop = config["distanceStop"]
        self.distance_slow = config["distanceSlow"]


    def get_direction(self) -> int:
        """Returns the current driving direction. If the driving time is over this returns stop.

        Returns:
            int: Driving direction
        """        
        if time.time() > self.stop_timer:
            return Directions.stop
        else:
            return self.direction

    def set_direction(self, direction: int) -> None:
        """Sets the current driving direction and sets the driving timer accordingly.

        Args:
            direction (int): Driving direction
        """        
        self.direction = direction
        if (direction == Directions.turn_left or
            direction == Directions.turn_right):
            self.stop_timer = time.time() + self.driving_time_turning
        else:
            self.stop_timer = time.time() + self.driving_time

    def get_speed(self) -> float:
        """Returns the driving speed. If there is an obstacle in front of the car
        a slower speed is returned. If the obstacle is too close a speed of zero is returned 
        to stop the car.

        Returns:
            float: Speed
        """   
        if self.distance < self.distance_stop:
            print("STOP: Obstacle detected ({} cm)".format(self.distance))
            return 0
        elif self.distance < self.distance_slow:     
            return self.speed * 0.8
        else:
            return self.speed

    def reset_stop_timer(self) -> None:
        """Resets the stop timer to 0
        """        
        self.stop_timer = 0


class Directions(IntEnum):
    """Maps the driving directions to integer values
    """    
    forward = 1
    left = 2
    reverse = 3
    right = 4
    stop = 5
    turn_left = 6
    turn_right = 7

