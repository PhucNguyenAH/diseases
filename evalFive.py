import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from models import Five
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report

class2idx = {
    "CRC":0,
    "Lung":1,
    "Gastric":2,
    "Liver":3,
    "Breast":4
}

idx2class = {v: k for k, v in class2idx.items()}

testloader = []
checkpoint_path="checkpoints/ckp_five.pt"
checkpoint_last_path="checkpoints/ckp_five_last.pt"

with open('data/val_five.npy', 'rb') as f:
    x_test0 = np.load(f)
with open('data/val_five_target.npy', 'rb') as f:
    y_test0 = np.load(f)

with open('data/test_five.npy', 'rb') as f:
    x_test1 = np.load(f)
with open('data/test_five_target.npy', 'rb') as f:
    y_test1 = np.load(f)

x_test = np.concatenate([x_test0, x_test1])
y_test = np.concatenate([y_test0, y_test1])

x_test = np.array(x_test, np.float32)

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


test_dataset = ClassifierDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

testloader = DataLoader(dataset=test_dataset, batch_size=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Five()
checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
print("f1 score: ", checkpoint_dict['f1'])
model.load_state_dict(checkpoint_dict['net'])
model.to(device)

y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in testloader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
fig = sns.heatmap(confusion_matrix_df, annot=True)
fig.get_figure().savefig("confusion_matrix_five.jpg")
print(classification_report(y_test, y_pred_list))
