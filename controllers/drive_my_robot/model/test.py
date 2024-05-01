from data_manager import *

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch import optim
import matplotlib.pyplot as plt

from unet_model import UNet


if __name__ == '__main__':
    path = Path(__file__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = UNet(3,1)
    model.load_state_dict(torch.load(str(path.parent.absolute()) + "\\model.pth"))
    model.to(device)
    
    test_ds = DepthDataset("test")
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        model.eval()
        test_loss = 0
        counter = 0
        for x,y in tqdm(test_dl):
            y = y.unsqueeze(1)
            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            x,y = x.to(device), y.to(device, dtype=mask_type)
            

            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

            if(counter == 0):
                plt.subplot(3, 1, 1)
                plt.axis("off")
                plt.imshow(x[0][0].cpu().numpy())
                plt.subplot(3, 1, 2)
                plt.axis("off")
                plt.imshow(pred[0][0].detach().cpu().numpy(),cmap=plt.get_cmap('Spectral_r'))
                plt.subplot(3, 1, 3)
                plt.axis("off")
                plt.imshow(y[0][0].detach().cpu().numpy(),cmap=plt.get_cmap('Spectral_r'))
                plt.show()

            counter+=1


        print("----------TESTING COMPLETE----------")
        print(f"Test loss: {test_loss}")

            


