import torch
import pandas as pd
from tqdm import tqdm
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, BrainDataset, EmbedSet
from torch.utils.data import Dataset, DataLoader
from simclr import Classifier


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

classify = Classifier(embed_dim=1000, hidden_dim=12, out_class=2)
optimizer = torch.optim.Adam(classify.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classify.to(device)


annotations_file = pd.read_csv('./annotations_Train_PPMIs.csv', header=None)
annotations_val = pd.read_csv('./annotations_Val_PPMIs.csv', header=None)
annotations_test = pd.read_csv('./annotations_Test_PPMIs.csv', header=None)
batch_size=32
epochs = 800
logfile = './simclr_clf_logs.txt'
logsval = './simclr_clf_val.txt'
logstest = './simclr_clf_test.txt'

trainset = EmbedSet(annotations_file, device)
trainloader = DataLoader(trainset, batch_size, shuffle=True)
valset = EmbedSet(annotations_val, device)
valloader = DataLoader(valset, 8, shuffle=True)
testset = EmbedSet(annotations_test, device)
testloader = DataLoader(testset, 8, shuffle=True)
with open (logfile, 'w') as logs:
    logs.write(f'This is a classification experiment on the PPMI dataset with finetuned simclr model \n\n')
with open (logsval, 'w') as logs:
    logs.write(f'This contains val results for classification experiment on the PPMI dataset with finetuned simclr model \n\n')
with open (logstest, 'w') as logs:
    logs.write(f'This contains test results for classification experiment on the PPMI dataset with finetuned simclr model \n\n')

print(sum([x.reshape(-1).shape[0] for x in classify.parameters()]))



def test_clf(test_set, logfile):
    classify.eval()
    running_loss=0.0
    running_acc=.0
    running_TP, running_FP, running_FN = 0, 0, 0
    for example, label in tqdm(test_set, desc=f'test_epoch {epoch+1}', ncols=70, leave=False):
        # repr = byol(example)          # repr.shape = 512
        pred = classify(example)       # pred.shape= batch_size x 2
        label = label.to(device)
        loss = loss_fn(pred, label)
        running_loss+=loss
        pred_label = torch.argmax(pred, dim=1)
        running_acc+=(pred_label == label).sum().item()
        running_TP += ((pred_label==1) & (label==1)).sum().item()
        running_FP += ((pred_label==1) & (label==0)).sum().item()
        running_FN += ((pred_label==0) & (label==1)).sum().item()
    running_acc=running_acc/len(annotations_test)
    running_loss=running_loss/len(test_set)
    with open (logfile, 'a') as logs:
        logs.write(f'Epoch [{epoch+1:>3}] \t\t Loss: {running_loss:2.5f} \t\t Accuracy: {running_acc:2.5f} \t\t Precision: {(running_TP/(running_TP+running_FP+1e-8)):2.5f} \t\t Recall: {(running_TP/(running_TP+running_FN+1e-8)):2.5f} \n')
    classify.train()
    return
   
min_loss = float('inf')
# min_loss = 0.05130
for epoch in range(epochs):
    running_loss=0.0
    running_acc=.0
    running_TP, running_FP, running_FN = 0, 0, 0
    for example, label in tqdm(trainloader, desc=f'epoch {epoch+1}', ncols=70, leave=False):
        # repr = byol(viewo)          # repr.shape = 512
        pred = classify(example)       # pred.shape= batch_size x 2
        label = label.to(device)
        loss = loss_fn(pred, label)
        running_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred_label = torch.argmax(pred, dim=1)
        running_acc+=(pred_label == label).sum().item()
        running_TP += ((pred_label==1) & (label==1)).sum().item()
        running_FP += ((pred_label==1) & (label==0)).sum().item()
        running_FN += ((pred_label==0) & (label==1)).sum().item()
    running_acc=running_acc/len(annotations_file)
    running_loss=running_loss/len(trainloader)
    # state = open(logfile, 'a')
    # state.write(f'Epoch [{epoch+1:>3}] \t\t Loss: {running_loss:2.5f}')
    # state.close()
    with open (logfile, 'a') as logs:
        logs.write(f'Epoch [{epoch+1:>3}] \t\t Loss: {running_loss:2.5f} \t\t Accuracy: {running_acc:2.5f} \t\t Precision: {(running_TP/(running_TP+running_FP+1e-8)):2.5f} \t\t Recall: {(running_TP/(running_TP+running_FN+1e-8)):2.5f} \n')
    if running_loss<min_loss and epoch>4:
        min_loss=running_loss
        torch.save(classify.state_dict(), f'Classif_state_f.pth')
        # torch.save(byol, f'Classif_state_All.pth')
        with open (logfile, 'a') as logs:
            logs.write(f'Model saved at epoch {epoch}\n')
    test_clf(valloader, logsval)
    test_clf(testloader, logstest)

print("Done training!")