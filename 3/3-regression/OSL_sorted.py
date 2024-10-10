import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def x_matrix(x: np.ndarray, n) -> np.ndarray:
    """
    Create Polynomial Features
    """
    n_samples = x.shape[0]
    x_temp = np.ones((n_samples, 1))
    
    for deg in range(1, n + 1):
        x_temp = np.hstack((x_temp, x ** deg))
    
    return x_temp


def estimate_b(x, y):
    """
    Estimate Parameters Using OLS
    """
    b = np.linalg.inv(x.T @ x) @ (x.T @ y)
    return b


# Load data
data = pd.read_csv('train.csv')
train_x = data['x'].to_numpy()
train_y = data['y'].to_numpy()


# Get the indices that would sort x
sorted_indices = np.argsort(train_x) 
x_sorted = train_x[sorted_indices]   
y_sorted = train_y[sorted_indices]

# Make Input Matrix
degree = 21
X_train = x_matrix(x_sorted.reshape(-1, 1), degree)

# Estimate Parameters For Training Data
B = estimate_b(X_train, y_sorted)

# Predict Outputs
pred_y_train = X_train @ B

# Error
ssr_train = np.sum((pred_y_train - y_sorted) ** 2)
print(f'SSR: {ssr_train}')

# Save the parameters to a file
with open('weights.pkl', 'wb') as f:
    pickle.dump(B, f)

# Load test data
test_data = pd.read_csv('test.csv')  
test_x = test_data[['x']].to_numpy()
test_id = test_data['id'].to_numpy()

# Create polynomial features for test data
X_test = x_matrix(test_x, degree)

# Make predictions on the test set
test_predictions = X_test @ B

# Save test predictions to CSV
test_output = pd.DataFrame({'id': test_id, 'x': test_x.flatten(), 'y': test_predictions.flatten()})
test_output.to_csv('test_predictions.csv', index=False)

# Plotting
plt.figure(dpi=300)
plt.rcParams['font.family'] = 'Cambria'
plt.scatter(x_sorted, y_sorted, color='salmon', label='Training Data', edgecolor='black', s=20)
plt.plot(x_sorted, pred_y_train, color='blue', label='Polynomial Fit (Train)', linewidth=2)
plt.title('Polynomial Regression Fit', fontsize=14, fontweight='bold')
plt.xlabel('X-axis', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
plt.show()
