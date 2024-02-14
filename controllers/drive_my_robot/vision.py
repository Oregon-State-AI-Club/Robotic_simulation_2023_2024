from controller import Camera
import numpy as np
import cv2

class vision:
    
    def __init__(self,camera,timestep,height,width):
        
        self.image = None
        
        self.camera = camera
        self.timestep = timestep
        self.height = height
        self.width = width
        
        cv2.startWindowThread()
        cv2.namedWindow("preview")
        
        
        
    def contourImage(self):
        imgray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgray, contours, -1, (0,255,0), 3)
        self.image = imgray
        
        
    def grabImage(self):
        cameraData = self.camera.getImage();
        self.image = np.frombuffer(cameraData, np.uint8).reshape((self.height,self.width, 4))
        
    def displayImage(self):

        cv2.imshow("preview", self.image)
        cv2.waitKey(self.timestep)
