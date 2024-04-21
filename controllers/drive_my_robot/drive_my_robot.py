
from controller import Camera
from controller import Display
from controller import Supervisor

import numpy as np
import cv2
import random


from vision import vision


robot = Supervisor()

zones = [[-0.709,-2.12],[-0.67,3.05]]

timestep = int(robot.getBasicTimeStep())
max_speed = 6.28

#These are the name properties of the rotational motor objects
#of the robot
lidar = robot.getDevice("lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

camera1 = robot.getDevice("camera1");
camera1.enable(timestep)


display1 = robot.getDevice('display1');


robot_node = robot.getFromDef("robot")

wheels = ["motor_1","motor_2","motor_3","motor_4"]
impactSensors = ["front",
                 "rear"]

#For each of the named motors, set default position and velocity
for wheel in wheels:
    tempWheel = robot.getDevice(wheel)
    tempWheel.setPosition(float('inf'))
    tempWheel.setVelocity(0.0)
    
for sensor in impactSensors:
    impactSensor = robot.getDevice(sensor)
    impactSensor.enable(timestep)
    
    
width = camera1.getWidth()
height = camera1.getHeight()

camFeed = vision(camera1,timestep,width,height)

#While the simulation is running
while robot.step(timestep) != -1:
    if(int(robot.getTime())%18000==0 ):
        
        robot.simulationResetPhysics()
    
    camFeed.grabImage()
    #camFeed.contourImage()
    #camFeed.displayImage()
    #Set all the motors to a constaint speed
    #Since all of the motors have the same orientation, this 
    #results in a straight line
    
    range_image = lidar.getRangeImage()
    #print("{}".format(range_image))
    
    left_speed = 0.5*max_speed
    for wheel in wheels:
        tempWheel = robot.getDevice(wheel)
        tempWheel.setVelocity(left_speed)
        
    
    for sensor in impactSensors:
        impactSensor = robot.getDevice(sensor)
        if(impactSensor.getValue()):
            translation_field = robot_node.getField('translation')
            zone = random.choice(zones)
            deltaX = random.randrange(0,12,1)/10.0
            deltaY = -1*random.randrange(0,12,1)/10.0
            newTranslation = [zone[0]+deltaX, zone[1]+deltaY, -0.120231]
            translation_field.setSFVec3f(newTranslation)
            
            """
            rotation_field = robot_node.getField('rotation')
            curRotation = rotation_field.getSFVec3f()
            changeYZ = random.randrange(0,70,1)/100.0
            newRotation = [curRotation[0],curRotation[1],changeYZ,changeYZ]
            rotation_field.setSFRotation(newRotation)
            """
        
        #AFTER COLLISION, NEED TO SEMI-RANDOMLY
        #RELOCATE ROBOT POSITION AND ORIENTATION
        
        
    
    
    