import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score




if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

with open ('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

train = pd.read_csv(cfg['selary']['train_refactor'])
test = pd.read_csv(cfg['selary']['test_refactor'])

Y_train = train['Credit_Score']
X_train = train.drop('Credit_Score', axis = 1)

Y_test = train['Credit_Score']
X_test = train.drop('Credit_Score', axis = 1)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)


train_t = TensorDataset(X_train, Y_train)
test_t = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_t, batch_size=64, shuffle=True)
test_loader = DataLoader(test_t, batch_size=64, shuffle=True)

import torch.optim as optim
class CreditScore(nn.Module):
    def __init__(self, input_size):
        super(CreditScore, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

size = X_train.shape[1]
model = CreditScore(size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 60 

for epch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        #print( labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()  * inputs.size(0)
        
        
    ephoch_loss = running_loss / len(train_loader)
    print(ephoch_loss)

model.eval()
predicted = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predict = torch.max(outputs, 1)
        predicted.extend(predict.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
accurancy = accuracy_score(true_labels, predicted)
print(accurancy)
