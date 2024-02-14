from controller import Robot
from controller import Camera
from controller import Display

import numpy as np
import cv2

from vision import vision


robot = Robot()

timestep = int(robot.getBasicTimeStep())
max_speed = 6.28

#These are the name properties of the rotational motor objects
#of the robot
camera = robot.getDevice("camera");
camera.enable(timestep)



wheels = ["motor_1","motor_2","motor_3","motor_4"]
#For each of the named motors, set default position and velocity
for wheel in wheels:
    tempWheel = robot.getDevice(wheel)
    tempWheel.setPosition(float('inf'))
    tempWheel.setVelocity(0.0)
    
width = camera.getWidth()
height = camera.getHeight()

camFeed = vision(camera,timestep,width,height)

#While the simulation is running
while robot.step(timestep) != -1:
    camFeed.grabImage()
    camFeed.contourImage()
    camFeed.displayImage()
    #Set all the motors to a constaint speed
    #Since all of the motors have the same orientation, this 
    #results in a straight line
    
    left_speed = 0.5*max_speed
    for wheel in wheels:
        tempWheel = robot.getDevice(wheel)
        tempWheel.setVelocity(left_speed)
        
    
    
    