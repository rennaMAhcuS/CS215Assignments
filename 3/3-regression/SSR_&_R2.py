import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Create polynomial features
def X_matrix(X, N):
    n_samples = X.shape[0]
    X_temp = np.ones((n_samples, 1))
    for deg in range(1, N + 1):
        X_temp = np.hstack((X_temp, X ** deg))
    return X_temp


# Estimate parameters using OLS
def estimate_B(X, Y):
    B = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    return B


# Load data
data = pd.read_csv('train.csv')
train_x = data['x'].to_numpy()
train_y = data['y'].to_numpy()

# Set seed for reproducibility
np.random.seed(60)

# Randomly shuffle indices
indices = np.arange(len(train_x))
np.random.shuffle(indices)

# Split data into training and dev sets (90:10)
split_index = int(len(train_x) * 0.9)
train_indices = indices[:split_index]
dev_indices = indices[split_index:]

# Create training and dev sets
train_x, dev_x = train_x[train_indices], train_x[dev_indices]
train_y, dev_y = train_y[train_indices], train_y[dev_indices]

degrees = range(1, 23)
train_ssr = []
dev_ssr = []
train_r2 = []
dev_r2 = []

# Fit models for each degree
for degree in degrees:
    X_train = X_matrix(train_x.reshape(-1, 1), degree)
    B = estimate_B(X_train, train_y)
    
    # Predictions for training and dev sets
    pred_train = X_train @ B
    X_dev = X_matrix(dev_x.reshape(-1, 1), degree)
    pred_dev = X_dev @ B
    
    # Calculate SSR
    train_ssr.append(np.sum((pred_train - train_y) ** 2))
    dev_ssr.append(np.sum((pred_dev - dev_y) ** 2))

    # Calculate SSY
    SSy_train = np.sum((train_y - np.mean(train_y)) ** 2)
    SSy_dev = np.sum((dev_y - np.mean(dev_y)) ** 2)

    # Calculate R^2
    train_r2.append( 1 - (np.sum((pred_train - train_y) ** 2) / SSy_train))
    dev_r2.append( 1 - (np.sum((pred_dev - dev_y) ** 2) / SSy_dev))

# Plotting Training and Dev SSR
print(train_ssr,dev_ssr)
diff = np.abs(np.array(train_ssr) - np.array(dev_ssr))
print(diff)

plt.plot(degrees, diff)
plt.title('Difference between train_SSr & dev_SSr')
plt.show(block=False)

plt.figure(figsize=(18, 6))
plt.subplot(1,2,1)
plt.plot(degrees, train_ssr, label = 'Training Error', marker = 'o', color = 'blue')
plt.plot(degrees, dev_ssr, label = 'Development Error', marker = 'o', color = 'red')

plt.xlabel('Polynomial Degree')
plt.ylabel('Error')
plt.title('SSR for Degrees')
plt.legend()
plt.grid(True)
plt.yscale('log')

plt.subplot(1,2,2)
plt.plot(degrees, train_r2, label = 'train_R^2', marker = 'o', color = 'blue')
plt.plot(degrees, dev_r2, label = 'dev_R^2', marker = 'o', color = 'red')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2')
plt.title('R^2 for Degrees')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
