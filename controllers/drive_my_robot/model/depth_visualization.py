import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from pathlib import Path
SIZE = 480,140

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    im = Image.open(filename)
    im.resize(SIZE, Image.Resampling.LANCZOS) 
    depth_png = np.array(im, dtype=int) 
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return depth


path = Path(__file__)
print("File Path: ", str(path.parent.absolute())+"\\data\\depth_selection\\test_depth_completion_anonymous\\velodyne_raw\\")



for i in range(10):
  depthPath = str(path.parent.absolute())+"\\data\\depth_selection\\test_depth_completion_anonymous\\velodyne_raw\\"
  imagePath = str(path.parent.absolute())+"\\data\\depth_selection\\test_depth_completion_anonymous\\image\\"
  depthPath += "000000000"+str(i)+".png"
  imagePath += "000000000"+str(i)+".png"

  im = Image.open(depthPath)
  im = im.resize(SIZE, Image.Resampling.LANCZOS)
  plt.subplot(3, 1, 1)
  plt.axis("off")
  plt.imshow(im)
  plt.subplot(3, 1, 2)
  plt.axis("off")
  plt.imshow(depth_read(depthPath),cmap=plt.get_cmap('Spectral_r'))
  plt.subplot(3,1,3)
  plt.axis("off")
  plt.imshow(Image.open(imagePath))
 
  plt.show()