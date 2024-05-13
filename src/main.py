import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from DataPreProcessor import DataPreProcessor
from Actuator import Actuator
from training import compute_metrics, training
from validating import validating
from evaluating import evaluating
import matplotlib.pyplot as plt
from torch import nn, optim

# Load data using DataPreProcessor
data_prep = DataPreProcessor('data_set.csv')
train_loader, val_loader = data_prep.load_data()

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Actuator().to(device)

# Define loss function, optimizer, and learning rate scheduler
criterion = nn.SmoothL1Loss(beta=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model
epochs = 30
metrics = {'train_loss': [], 'val_loss': [], 'train_rmse': [], 'val_rmse': []}

for epoch in range(epochs):
    train_loss, train_rmse = training(model, train_loader, criterion, optimizer, device)
    val_loss, val_rmse = validating(model, val_loader, criterion, device)

    metrics['train_loss'].append(train_loss)
    metrics['val_loss'].append(val_loss)
    metrics['train_rmse'].append(train_rmse)
    metrics['val_rmse'].append(val_rmse)

    print(f'Epoch {epoch+1}')

# Plot training and validation metrics
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Metrics over Epochs') 

axs[0].plot(metrics['train_loss'], label='Training Loss')
axs[0].plot(metrics['val_loss'], label='Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].grid()
axs[0].legend()

axs[1].plot(metrics['train_rmse'], label='Training RMSE')
axs[1].plot(metrics['val_rmse'], label='Validation RMSE')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('RMSE')
axs[1].grid()
axs[1].legend()

plt.show()

# Save the trained model
model_save_path = 'model.pth'
torch.save(model.state_dict(), model_save_path)

# Evaluate the model
csv_path = 'inference.csv'
model_path = 'model.pth'
evaluating(csv_path, device, model_path)
