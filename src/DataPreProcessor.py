import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

class DataPreProcessor:

    def __init__(self, file_path, batch_size=64, train_split=0.75):
        self.file_path = file_path
        self.batch_size = batch_size
        self.train_split = train_split

    def load_data(self):
        data = pd.read_csv(self.file_path)
        X = data[['Position error', 'Velocity']].values
        y = data['Torque'].values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)

        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
