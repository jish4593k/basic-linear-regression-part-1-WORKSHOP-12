import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define the linear regression model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predictions
y_pred = model(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred.detach().numpy())

# Visualize the Training set results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train.numpy(), y_train.numpy(), color='red', label='Actual')
plt.plot(X_train.numpy(), model(X_train).detach().numpy(), color='blue', label='Predicted')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

# Visualize the Test set results
plt.subplot(1, 2, 2)
plt.scatter(X_test.numpy(), y_test.numpy(), color='red', label='Actual')
plt.plot(X_test.numpy(), y_pred.detach().numpy(), color='blue', label='Predicted')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

# Additional data visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X.numpy(), y, color='red')
plt.plot(X_train.numpy(), model(X_train).detach().numpy(), color='blue')
plt.title('Salary vs Experience (Full Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# Plot loss vs. epoch
losses = []
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    losses.append(loss.item())

plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), losses)
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()

print(f"Mean Squared Error: {mse:.2f}")
