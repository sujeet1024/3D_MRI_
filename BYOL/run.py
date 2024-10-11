import torch
import pandas as pd
from BYOL.byol import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from BYOL.utils import BrainDataset
from tqdm import tqdm

# Initialize seed and hyperparameters
seed = 0
imgSize = 48 #256

# Ensure reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Initialize backbone, BYOL and optimizer
resnet = models.video.r3d_18(pretrained=False)
resnet.stem[0] = torch.nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
byol = BYOL(resnet, imageSize=imgSize, embeddingLayer='avgpool')
optimizer = torch.optim.Adam(byol.parameters(), lr=3e-4)

# GPU compatibility 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# byol = torch.load('./BYOL_state_ADMIC.pth')
byol.load_state_dict(torch.load('./BYOL_state.pth'), strict=False)
# state = torch.load('./BYOL_state_ADMIC.pth')
# byol.load_state_dict(state)
byol.to(device)
byol.train()
annotations_file = pd.read_csv('../train_PPMIs.csv', header=None)
batch_size=32
epochs = 50
logfile = './byol_logs3.txt'

trainset = BrainDataset(annotations_file, device)
trainloader = DataLoader(trainset, batch_size, shuffle=True)
with open (logfile, 'w') as logs:
    logs.write(f'This experiment produces BYOL model tuned for PPMI dataset: BYOL_state_F saves the model state, and BYOL_stat_All saves all the model components \n\n')

# Train embedding model according to BYOL paper
min_loss = float('inf')
# min_loss = 0.05130
for epoch in range(epochs):
    running_loss=0.0
    for example, label in tqdm(trainloader, desc=f'epoch {epoch+1}', ncols=70, leave=False):
        loss = byol.contrast(example, label)
        running_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    running_loss=running_loss/len(trainloader)
    byol.updateTargetEncoder()
    # state = open(logfile, 'a')
    # state.write(f'Epoch [{epoch+1:>3}] \t\t Loss: {running_loss:2.5f}')
    # state.close()
    with open (logfile, 'a') as logs:
        logs.write(f'Epoch [{epoch+1:>3}] \t\t Loss: {running_loss:2.5f}\n')
    if running_loss<min_loss and epoch>4:
        min_loss=running_loss
        torch.save(byol.state_dict(), f'BYOL_state_F.pth')
        torch.save(byol, f'BYOL_state_All.pth')
        with open (logfile, 'a') as logs:
            logs.write(f'Model saved at epoch {epoch}\n')
    if running_loss<=0.028:
        break
    # images = torch.randn(10, 1, imgSize, imgSize, imgSize).to(device)
    # loss = byol.contrast(images)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # byol.updateTargetEncoder() # update target encoder by EMA
    # print(f'Epoch {epoch+1:>2} --- Loss: {loss.item():2.5f}')
print("Done training!")



# images = torch.randn(10, 1, imgSize, imgSize, imgSize) #.to(device)
# embeddings = byol(images)
# print(embeddings.size())
