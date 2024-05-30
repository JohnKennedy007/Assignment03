import torch
import torch.nn as nn
import torch.optim as optim

class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def fit_regression_model(X, y, learning_rate=0.01, epochs=5000):
    input_dim = X.shape[1]
    model = RegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1000 == 0:
            print(f'Epoch {epoch+1}: loss = {loss.item()}')

    return model, loss.item()

def get_train_data(dim=1):
    if dim == 1:
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
        y = torch.tensor([[7.0], [9.0], [11.0], [13.0], [15.0]], dtype=torch.float32)
    else:
        X = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=torch.float32)
        y = torch.tensor([[6.0], [8.0], [10.0], [12.0], [14.0]], dtype=torch.float32)
    
    return X, y
