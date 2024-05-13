def compute_metrics(outputs, targets):
    mse = nn.MSELoss()(outputs, targets)
    rmse = torch.sqrt(mse)
    return mse.item(), rmse.item()