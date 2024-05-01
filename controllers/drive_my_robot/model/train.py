from data_manager import *

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt

from unet_model import UNet

torch.cuda.empty_cache()
LR = 1e-5
EPOCHS = 10

IMG_HEIGHT = 352
IMG_WIDTH = 1216



if __name__ == '__main__':
    path = Path(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = UNet(3,1).to(device)
    
    train_ds = DepthDataset("train")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    

    loss_fn = nn.MSELoss()
    opt = optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
    epochCount = 0
    total_train_loss = 0
    
    for epoch in range(EPOCHS):
        
        epoch_train_loss = 0
        model.train()
        epochCount+=1
        print("Epoch: ",epochCount)
        counter = 0
        for x,y in tqdm(train_dl):
            y = y.unsqueeze(1)
            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            x,y = x.to(device), y.to(device, dtype=mask_type)

            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            counter+=1
            epoch_train_loss += loss.item()
            
        total_train_loss += epoch_train_loss
        print("Epoch loss: ", epoch_train_loss)
        print("Total loss: ", total_train_loss)
    print("----------TRAINING COMPLETE----------")
    torch.save(model.state_dict(), str(path.parent.absolute()) + "\\model.pth")
    
        