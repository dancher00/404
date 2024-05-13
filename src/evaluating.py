import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def evaluating(csv_path, device, model_path):
    data = pd.read_csv(csv_path)
    X = data[['Position error', 'Velocity']].values
    y_true = data['Torque'].values
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Network = Actuator().to(device)
    Network.load_state_dict(torch.load(model_path))
    Network.eval()
    
    with torch.no_grad():
        y_pred_tensor = Network(X_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted by Network', linestyle='--')
    plt.title('Comparison of Actual and Predicted Torque')
    plt.xlabel('Sample')
    plt.ylabel('Torque')
    plt.legend()
    plt.text(0.05, 0.95, 'MAE: {:.3f}\nRMSE: {:.3f}'.format(mae, rmse), 
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.grid()
    plt.show()
