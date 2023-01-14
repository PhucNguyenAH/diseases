import pandas as pd
from sklearn import preprocessing
import numpy as np
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

with open('data/train_bin.npy', 'rb') as f:
    x_train = np.load(f)
with open('data/train_bin_target.npy', 'rb') as f:
    y_train = np.load(f)

with open('data/val_bin.npy', 'rb') as f:
    x_test = np.load(f)
with open('data/val_bin_target.npy', 'rb') as f:
    y_test = np.load(f)

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# x_train, x_test, y_train, y_test = torch.from_numpy(x_train), torch.from_numpy(x_test), torch.from_numpy(y_train).type(torch.LongTensor), torch.from_numpy(y_test).type(torch.LongTensor)

model_file_path = "checkpoints/ckp_bin.pt"
checkpoint_last_path="checkpoints/ckp_bin_last.pt"

train_acc_MAX=0
# val_acc_MAX = 0
best_f1_score = 0
EPOCHS =10000
batch_size = 246
trainloader = []
testloader = []

# accuracy_stats = {
#     'train': [],
#     "val": []
# }
loss_stats = {
    'train': [],
    "val": []
}

f1_score_stats = []

num_features = 8630
n_hidden_1 = 8192
# n_hidden_2 = 2048
# n_hidden_3 = 512
# n_hidden_3 = 512
# n_hidden_4 = 128
# n_hidden_5 = 32

num_classes = 2

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

# for (i,j) in zip(x_train, y_train):
#     trainloader.append([i,j])
# trainloader = torch.utils.data.DataLoader(trainloader, shuffle=True, batch_size=batch_size)

# for (i,j) in zip(x_test, y_test):
#     testloader.append([i,j])
# testloader = torch.utils.data.DataLoader(testloader, shuffle=True, batch_size=batch_size)

class AutoscaleFocalLoss:
    def __init__(self, threshold):
        self.threshold = threshold
    
    def gamma(self, logits):
        return self.threshold/2 * (torch.cos(np.pi*(logits+1)) + 1)

    def __call__(self, logits, labels):
        labels = F.one_hot(labels, 2)
        
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

# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_ff=8192, dropout = 0.1):
#         super().__init__() 
#         # We set d_ff as a default to 8192
#         self.linear_1 = nn.Linear(d_model, d_ff)
#         self.dropout = nn.Dropout(dropout)
#         self.linear_2 = nn.Linear(d_ff, d_model)
#     def forward(self, x):
#         x = self.dropout(F.relu(self.linear_1(x)))
#         x = self.linear_2(x)
#         return x

# class Norm(nn.Module):
#     def __init__(self, d_model, eps = 1e-6):
#         super().__init__()
    
#         self.size = d_model
#         # create two learnable parameters to calibrate normalisation
#         self.alpha = nn.Parameter(torch.ones(self.size))
#         self.bias = nn.Parameter(torch.zeros(self.size))
#         self.eps = eps
#     def forward(self, x):
#         norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
#         / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
#         return norm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_features, n_hidden_1)
        # self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        # self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        # self.fc4 = nn.Linear(n_hidden_3, n_hidden_4)
        # self.fc5 = nn.Linear(n_hidden_4, n_hidden_5)
        # self.fc6 = nn.Linear(n_hidden_5, n_hidden_6)
        self.layer_out = nn.Linear(n_hidden_1, num_classes)

        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden_1)
        # self.batchnorm2 = nn.BatchNorm1d(n_hidden_2)
        # self.batchnorm3 = nn.BatchNorm1d(n_hidden_3)
        # self.batchnorm4 = nn.BatchNorm1d(n_hidden_4)
        # self.batchnorm5 = nn.BatchNorm1d(n_hidden_5)
        # self.batchnorm6 = nn.BatchNorm1d(n_hidden_6)
        # self.fc = nn.Linear(num_features, n_hidden)
        # self.norm = Norm(n_hidden)
        # self.ff = FeedForward(n_hidden)
        # self.dropout = nn.Dropout(p=0.1)
        # self.layer_out = nn.Linear(n_hidden2, num_classes)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.fc1(x)))
        # x = self.dropout(F.relu(self.batchnorm2(self.fc2(x))))
        # x = self.dropout(F.relu(self.batchnorm3(self.fc3(x))))
        # x = self.dropout(F.relu(self.batchnorm4(self.fc4(x))))
        # x = self.dropout(F.relu(self.batchnorm5(self.fc5(x))))
        # x = self.dropout(F.relu(self.batchnorm6(self.fc6(x))))

        # x = F.relu(self.batchnorm1(self.fc1(x)))
        # x2 = self.batchnorm2(x)
        # x = x + self.dropout(F.relu(self.fc2(x2)))
        # x2 = self.batchnorm3(x)
        # x = x + self.dropout(F.relu(self.fc3(x2)))
        # x2 = self.batchnorm4(x)
        # x = x + self.dropout(F.relu(self.fc4(x2)))
        # x2 = self.batchnorm5(x)
        # x = x + self.dropout(F.relu(self.fc5(x2)))

        # x = F.relu(self.batchnorm1(self.fc(x)))
        # x2 = self.norm(x)
        # x = x + self.dropout(self.ff(x2))
        # x2 = self.norm(x)
        # x = x + self.dropout(self.ff(x2))
        x = F.log_softmax(self.layer_out(x), -1)
        return x

use_cuda = torch.cuda.is_available()  #GPU cuda

net = Net()
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
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001, weight_decay=0.01, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000,14000,18000], gamma=0.1)

print(net)

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc



print("Begin training.")

last_loss = 1000
trigger_times = 0
<<<<<<< HEAD
patience = 5 
=======
patience = 5
>>>>>>> 3906940460ae85f2b3c467bb99f90926a1a70598
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
        # train_loss = criterion(outputs, labels)
        train_loss = calculate_preference_loss(outputs, labels)
        train_acc = multi_acc(outputs, labels)
        train_loss.backward()
        optimizer.step()
        # scheduler.step()
        train_epoch_loss += train_loss.item()
        # train_epoch_acc += train_acc.item()
        # # print statistics
        # running_loss += loss.item()
        # if i == len(trainloader)-1:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

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
            # val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            # val_epoch_acc += val_acc.item()
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    loss_stats['train'].append(train_epoch_loss/len(trainloader))
    loss_stats['val'].append(val_epoch_loss/len(testloader))
    f1 = f1_score(y_test, y_pred_list, average='macro')
    f1_score_stats.append(f1)  
    # accTrain_e = train_epoch_acc/len(trainloader)
    # accVal_e = val_epoch_acc/len(testloader)
    # accuracy_stats['train'].append(accTrain_e)
    # accuracy_stats['val'].append(accVal_e)

    # if val_acc_MAX < accVal_e or (val_acc_MAX == accVal_e and train_acc_MAX < accTrain_e):
    #     state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'accuracy': accVal_e}
    #     torch.save(state, model_file_path)
    #     train_acc_MAX = accTrain_e
    #     val_acc_MAX = accVal_e
    current_loss = val_epoch_loss/len(testloader)
    if f1 >= best_f1_score and current_loss < last_loss:
        best_f1_score = f1
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'f1': best_f1_score}
        torch.save(state, model_file_path)
    lr = optimizer.param_groups[0]['lr']
    # print(f'Epoch {epoch+0:03}: | LR: {lr:.5f} | Train Loss: {train_epoch_loss/len(trainloader):.5f} | Val Loss: {val_epoch_loss/len(testloader):.5f} | Train Acc: {train_epoch_acc/len(trainloader):.3f}| Val Acc: {val_epoch_acc/len(testloader):.3f}')
    if epoch%50==0:
        print(f'Epoch {epoch+0:04}: | LR: {lr:.5f} | Train Loss: {train_epoch_loss/len(trainloader):.5f} | Val Loss: {val_epoch_loss/len(testloader):.5f} | F1 score: {f1:.3f}')
<<<<<<< HEAD

=======
    
>>>>>>> 3906940460ae85f2b3c467bb99f90926a1a70598
    if current_loss > last_loss:
        last_loss = current_loss 
        trigger_times += 1
        print('Trigger Times:', trigger_times)

        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch}!')
            break
    else:
        trigger_times = 0
        last_loss = current_loss 
state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch, 'f1': f1}
torch.save(state, checkpoint_last_path)
print('Finished Training')
# Create dataframes
# train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,7))
# acc_plot = sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
loss_plot = sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')
loss_plot.get_figure().savefig("train_monitor.png")

# for i in range(len(accuracy_stats['val'])):
#     if val_acc_MAX < accuracy_stats['val'][i] or (val_acc_MAX == accuracy_stats['val'][i] and train_acc_MAX < accuracy_stats['train'][i]):
#         train_acc_MAX = accuracy_stats['train'][i]
#         val_acc_MAX = accuracy_stats['val'][i]

logger.info(f"best f1 score:{best_f1_score}")
# correct = 0
# total = 0
# # do đang thực hiện việc dự đoán nên ko cần tính đạo hàm
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         # chạy hàm dự đoán
#         outputs = net(images)
#         # the class với giá trị xác suất cao nhất là đâu ra dự đoán
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))