from data_manager import *

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt

from unet_model import UNet

cap = cv2.VideoCapture(0)
SIZE = 480,140


path = Path(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
model = UNet(3,1)
model.load_state_dict(torch.load(str(path.parent.absolute()) + "\\model.pth"))
model.to(device)

while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame = cv2.resize(frame, SIZE)
    # Display the resulting frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    img = torch.tensor(frame).permute(2,0,1).float()
    with torch.no_grad():
        img = img.unsqueeze(0)
        
        img = img.to(device)
        
        pred = model(img)
        cv2.imshow('frame',pred[0][0].detach().cpu().numpy())

    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()