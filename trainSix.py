import pandas as pd
from sklearn import preprocessing
import numpy as np
from models import Six
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import f1_score 

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import confusion_matrix, classification_report
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logging.txt",
    format="%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s",
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)

with open('data/train_six.npy', 'rb') as f:
    x_train = np.load(f)
with open('data/train_six_target.npy', 'rb') as f:
    y_train = np.load(f)

with open('data/val_six.npy', 'rb') as f:
    x_test = np.load(f)
with open('data/val_six_target.npy', 'rb') as f:
    y_test = np.load(f)

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

model_file_path = "checkpoints/ckp_six.pt"
checkpoint_last_path="checkpoints/ckp_six_last.pt"

train_acc_MAX=0
# val_acc_MAX = 0
best_f1_score = 0
EPOCHS = 10000
batch_size = 128
trainloader = []
testloader = []


loss_stats = {
    'train': [],
    "val": []
}

f1_score_stats = []

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
test_dataset = ClassifierDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

trainloader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size)
testloader = DataLoader(dataset=test_dataset, batch_size=1)

class AutoscaleFocalLoss:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def gamma(self, logits):
        return self.threshold/2 * (torch.cos(np.pi*(logits+1)) + 1)

    def __call__(self, logits, labels):
        labels = F.one_hot(labels, 6)
        assert logits.shape == labels.shape, \
                "Mismatch in shape, logits.shape: {} - labels.shape: {}".format(logits.shape, labels.shape)
        logits =  F.softmax(logits, dim=-1)
        CE = - labels * torch.log(logits)
        loss = ((1 - logits)**self.gamma(logits)) * CE
        loss = torch.sum(loss, dim=-1).mean()
        return loss


def calculate_preference_loss(logits, targets, mode='train'):

    ##Define loss function
    loss_function = AutoscaleFocalLoss(threshold = 2)
    
    #targets shape: [batch_size,]
    if mode == 'train':
        targets = targets.squeeze(-1)

    if torch.cuda.is_available():
        targets = targets.to(torch.long).cuda()
        logits = logits.cuda()

    loss = loss_function(logits = logits, labels = targets)

    return loss

use_cuda = torch.cuda.is_available()  #GPU cuda

net = Six()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
print(device)
net.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay= 1e-4)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.00001, weight_decay=0.01, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)

print(net)

print("Begin training.")

for epoch in range(1, EPOCHS+1):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    y_pred_list = []
    net.train()

    for i, data in enumerate(trainloader, 0):
        # load input và labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print(labels.shape)
        # print(outputs.shape)
        train_loss = criterion(outputs, labels)
        # train_loss = calculate_preference_loss(outputs, labels)
        train_loss.backward()
        optimizer.step()
        # scheduler.step()
        train_epoch_loss += train_loss.item()

        # VALIDATION    
    with torch.no_grad():
    
        val_epoch_loss = 0
        # val_epoch_acc = 0
        
        net.eval()
        for X_val_batch, y_val_batch in testloader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = net(X_val_batch)
                        
            # val_loss = criterion(y_val_pred, y_val_batch)
            _, y_pred_tags = torch.max(y_val_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            
            val_loss = calculate_preference_loss(y_val_pred, y_val_batch, mode='val')
            
            val_epoch_loss += val_loss.item()
            # val_epoch_acc += val_acc.item()
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    loss_stats['train'].append(train_epoch_loss/len(trainloader))
    loss_stats['val'].append(val_epoch_loss/len(testloader))
    f1 = f1_score(y_test, y_pred_list, average='macro')

    f1_score_stats.append(f1)  

    # current_loss = val_epoch_loss/len(testloader)
    if f1 >= best_f1_score:
        best_f1_score = f1
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'f1': best_f1_score}
        torch.save(state, model_file_path)
    lr = optimizer.param_groups[0]['lr']
    if epoch%20==0:
        print(f'Epoch {epoch+0:03}: | LR: {lr:.5f} | Train Loss: {train_epoch_loss/len(trainloader):.5f} | Val Loss: {val_epoch_loss/len(testloader):.5f} | F1 score: {f1:.3f}')

state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'f1': f1}
torch.save(state, checkpoint_last_path)
print('Finished Training')
# Create dataframes
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
loss_plot = sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
loss_plot.get_figure().savefig("train_monitor.png")


logger.info(f"best f1 score:{best_f1_score}")
