from controller import Robot

robot = Robot()

timestep = int(robot.getBasicTimeStep())
max_speed = 6.28

#These are the name properties of the rotational motor objects
#of the robot
wheels = ["motor_1","motor_2","motor_3","motor_4"]
#For each of the named motors, set default position and velocity
for wheel in wheels:
    tempWheel = robot.getDevice(wheel)
    tempWheel.setPosition(float('inf'))
    tempWheel.setVelocity(0.0)

#While the simulation is running
while robot.step(timestep) != -1:
    #Set all the motors to a constaint speed
    #Since all of the motors have the same orientation, this 
    #results in a straight line
    left_speed = 0.5*max_speed
    for wheel in wheels:
        tempWheel = robot.getDevice(wheel)
        tempWheel.setVelocity(left_speed)
    
    