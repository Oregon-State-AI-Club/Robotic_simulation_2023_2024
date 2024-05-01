import torch
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image 
from pathlib import Path
import cv2


BATCH_SIZE = 5 #No specific reason for this, medium recommended it lmao
SIZE = 608,176


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename).resize(SIZE), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.

    return torch.tensor(depth)

class DepthDataset(Dataset):
    
    def __init__(self,split):
        split = split.lower()
        path = Path(__file__)
        depthPath = ""
        imagePath = ""
        if(split == "test"):
            depthPath = str(path.parent.absolute()) + "\\data\\depth_selection\\val_selection_cropped\\velodyne_raw\\"
            imagePath = str(path.parent.absolute()) + "\\data\\depth_selection\\val_selection_cropped\\image\\"
        elif(split == "train"):
            depthPath = str(path.parent.absolute()) + "\\data\\depth_selection\\test_depth_completion_anonymous\\velodyne_raw\\"
            imagePath = str(path.parent.absolute()) + "\\data\\depth_selection\\test_depth_completion_anonymous\\image\\"
        else:
            print("Invalid split detected. 'test' or 'train' required, you input: " + split)
            exit()
        


        self.depthImgVector = []
        self.standardImgVector = []
        self.rgbdVector = []

        print("Loading and normalizing " + split + " depth images...")
        for depthImg in tqdm(os.listdir(depthPath)):
            normalizedDepth = depth_read(depthPath+depthImg)
            
            self.depthImgVector.append(normalizedDepth)
        

        print("Loading " + split + " RGB images...")
        for standardImg in tqdm(os.listdir(imagePath)):
            im = np.array(Image.open(imagePath+standardImg).resize(SIZE), dtype=int)
            #im.thumbnail(SIZE, Image.Resampling.LANCZOS)
            img = torch.tensor(im).permute(2,0,1).float()
            
            self.standardImgVector.append(img)
            
            
        

        print("Merging images...")
        for i in tqdm(range(len(self.depthImgVector))):
            self.rgbdVector.append([self.depthImgVector[i],self.standardImgVector[i]])
       
        print(split + " set loaded successfully\n")
        
    def __len__(self):
        return len(self.rgbdVector)
    
    def __getitem__(self, index):
        rgbImage = self.standardImgVector[index]
        depthImage = self.depthImgVector[index]
        return rgbImage,depthImage




"""
train_ds = DepthDataset("train")
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ds = DepthDataset("test")
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
"""