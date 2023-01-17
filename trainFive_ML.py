import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import pickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score 

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logging.txt",
    format="%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s:\t%(message)s",
    datefmt='%Y-%m-%d,%H:%M:%S',
    level=logging.INFO
)

with open('data/train_five.npy', 'rb') as f:
    x_train = np.load(f)
with open('data/train_five_target.npy', 'rb') as f:
    y_train = np.load(f)

with open('data/val_five.npy', 'rb') as f:
    x_test = np.load(f)
with open('data/val_five_target.npy', 'rb') as f:
    y_test = np.load(f)

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

svm_ = svm.SVC(kernel='linear', C=1.0, probability=True)
rf = RandomForestClassifier(criterion='gini', n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=3)

svm_.fit(x_train, y_train)
rf.fit(x_train, y_train)
knn.fit(x_train, y_train)

print("Accuracy:")
print(f'SVM: {accuracy_score(y_test, svm_.predict(x_test))}')
print(f'RF: {accuracy_score(y_test, rf.predict(x_test))}')
print(f'KNN: {accuracy_score(y_test, knn.predict(x_test))}')

print("F1 score:")
f1_svm = f1_score(y_test, svm_.predict(x_test), average='macro')
f1_rf = f1_score(y_test, rf.predict(x_test), average='macro')
f1_knn = f1_score(y_test, knn.predict(x_test), average='macro')
print(f'SVM: {f1_svm}')
print(f'RF: {f1_rf}')
print(f'KNN: {f1_knn}')

filename = 'checkpoints/ml_svm_five.sav'
pickle.dump(svm_, open(filename, 'wb'))
filename = 'checkpoints/ml_rf_five.sav'
pickle.dump(rf, open(filename, 'wb'))
filename = 'checkpoints/ml_knn_five.sav'
pickle.dump(knn, open(filename, 'wb'))