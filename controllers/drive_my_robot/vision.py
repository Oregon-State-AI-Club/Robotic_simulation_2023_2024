from controller import Camera
from controller import Display

import os 
import numpy as np
import cv2

class vision:
    
    def __init__(self,camera1,timestep,height,width):
        
        self.image1 = None

        
        self.camera1 = camera1

        
        self.timestep = timestep
        self.height = height
        self.width = width    
        cv2.startWindowThread()
        cv2.namedWindow("preview")  
        
    def contourImage(self):
        imgray = cv2.cvtColor(self.image1,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgray, contours, -1, (0,255,0), 2)
        self.image1 = imgray
        
        
    def grabImage(self):
        cameraData1 = self.camera1.getImage();
        self.image1 = np.frombuffer(cameraData1, np.uint8).reshape((self.height,self.width, 4))
       
    def displayImage(self):
        cv2.imshow("cam1", self.image1)
        cv2.waitKey(self.timestep)
        
        #cv2.imshow("cam2", self.image2)
        #cv2.waitKey(self.timestep)
    
    
    
    
   
    
       
