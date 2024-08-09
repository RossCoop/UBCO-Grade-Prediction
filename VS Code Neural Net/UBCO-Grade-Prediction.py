import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
import torch.nn.functional as F
from torch import flatten
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#### Functions

def df_to_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
  

    Y = df_as_np[:, 12]
  
    middle_matrix = np.delete(df_as_np, 12, axis=1)
    X = middle_matrix.reshape((len(Y), middle_matrix.shape[1], 1))
  


    return X, Y.astype(np.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Setting GPU
torch.set_default_device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#### Setting up input/output data
n = 10


## Loading df from csv
df = pd.read_csv('../../UBCO-Grade-Prediction-data/student-data-profs-factored.csv')
df = df.drop(df.columns[0], axis=1)
print(df)


X, y = df_to_X_y(df)

print(X)
print(y)

# Getting % of length of X/y
q_80 = int(len(y) * .8)
q_90 = int(len(y) * .9)

# Setting X, and y for training, validation, and testing
X_train, y_train = X[:q_80], y[:q_80]
X_val, y_val = X[q_80:q_90], y[q_80:q_90]
X_test, y_test = X[q_90:], y[q_90:]

# Saving train and test sets as tensors
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_val = torch.tensor(X_val).float()
y_val = torch.tensor(y_val).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Wrapping tensors with Dataset and then DataLoader
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, generator=torch.Generator(device='cuda'), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, generator=torch.Generator(device='cuda'), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, generator=torch.Generator(device='cuda'), shuffle=True)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

### MODEL BUILDING

class myCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # First Convolutional layer: 1 input channel, 8 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        # Second Convolutional layer: 8 input channels, 16 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layer: from 16*4*4 flattened to 16
        self.fc1 = nn.Linear(16 * 4 * 4, 16)
        
        # Final output layer to predict a single number
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # 1. Reshape to 4x4x1
        x = x.view(-1, 1, 4, 4)  # Reshape to [batch_size, channels, height, width]
        
        # 2. First CNN layer with ReLU activation
        x = F.relu(self.conv1(x))  # Output shape: [batch_size, 8, 4, 4]
        
        # 3. Second CNN layer with ReLU activation
        x = F.relu(self.conv2(x))  # Output shape: [batch_size, 16, 4, 4]
        
        # 4. Flatten the tensor
        x = x.view(-1, 16 * 4 * 4)  # Flatten to [batch_size, 16*4*4] = [batch_size, 256]
        
        # 5. Pass through the first fully connected layer with ReLU
        x = F.relu(self.fc1(x))  # Output shape: [batch_size, 16]
        
        # 6. Final output layer to get a single number
        x = self.fc2(x)  # Output shape: [batch_size, 1]
        
        return x

### Loss Function Building
class CustomLoss(nn.Module):
    def __init__(self, lambda_penalty=1.0):
        super(CustomLoss, self).__init__()
        self.lambda_penalty = lambda_penalty

    def forward(self, predictions, targets):
        # Calculate the MSE loss
        mse_loss = nn.MSELoss(predictions, targets)
        
        # Calculate the direction mismatch penalty
        direction_mismatch = ((predictions * targets) < 0).float()
        direction_penalty = self.lambda_penalty * (direction_mismatch.sum() / len(targets))
        
        # Combine MSE loss and direction penalty
        loss = mse_loss + direction_penalty
        return loss

## TO NOTE:
# changing the criterion is necessary as we need to heavily penalize price movement in the incorrect direction

### MODEL COMPILING

model = myCNN()
model.to(device)
print(model)


learning_rate = 0.001
num_epochs = 2
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


### MODEL RUNNING



# loop over our epochs
for e in range(num_epochs):
	# set the model in training mode
	model.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0
	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
	# loop over the training set
	for (x, y) in train_loader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))
		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = loss_function(pred, y)
		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()

with torch.no_grad():
        model.eval()
        # loop over the validation set
        for (x, y) in val_loader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += loss_function(pred, y)
            # calculate the number of correct predictions
            valCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        print('Val Loss: {0:.3f}'.format(totalValLoss))
        print('***************************************************')
        print()



# train_predictions = []
# y_train = []
# with torch.no_grad():
#     for batch_index, batch in enumerate(train_loader):
#         x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    
#         output = model(x_batch)
#         output = output.to('cpu').numpy()
#         train_predictions.append(output)
#         y_train.append(y_batch.to('cpu').numpy())




test_predictions = []
y_test = []
with torch.no_grad():
    model.eval()
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    
        output = model(x_batch)
        output = output.to('cpu').numpy()
        test_predictions.append(output)
        y_test.append(y_batch.to('cpu').numpy())
    
# Fixing array structure 
test_predictions = np.concatenate(test_predictions)


y_test = np.concatenate(y_test)

print(test_predictions)

plt.plot(y_test, label='Actual Grades')
plt.plot(test_predictions, label='Predicted Grades')
plt.xlabel('Day')
plt.ylabel('Grade %')
plt.legend()
plt.show()

exit()
predictions = pd.DataFrame({})
predictions["predictions"] = test_predictions
predictions["y_test"] = y_test
predictions["prediction bool"] = test_predictions > 0
predictions["y_test bool"] = y_test > 0

print(predictions)
predictions.to_csv("data/predictions.csv", index=False)