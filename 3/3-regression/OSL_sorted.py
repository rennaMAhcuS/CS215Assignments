import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def X_matrix(X: np.ndarray, N) -> np.ndarray:
    '''
    Create Polynomial Features
    '''
    n_samples = X.shape[0]
    X_temp = np.ones((n_samples, 1))
    
    for deg in range(1, N + 1):
        X_temp = np.hstack((X_temp, X ** deg))
    
    return X_temp


def estimate_B(X, Y):
    '''
    Estimate Parameters Using OLS
    '''
    B = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    return B


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
X_train = X_matrix(x_sorted.reshape(-1, 1), degree)

# Estimate Parameters For Training Data
B = estimate_B(X_train, y_sorted)

# Predict Outputs
pred_y_train = X_train @ B

# Error
ssr_train = np.sum((pred_y_train - y_sorted) ** 2)
print(f'SSR: {ssr_train}')

# Save the parameters to a file
with open('3_weights.pkl', 'wb') as f:
    pickle.dump(B, f)

# Load test data
test_data = pd.read_csv('test.csv')  
test_x = test_data[['x']].to_numpy()
test_id = test_data['id'].to_numpy()

# Create polynomial features for test data
X_test = X_matrix(test_x, degree)

# Make predictions on the test set
test_predictions = X_test @ B

# Save test predictions to CSV
test_output = pd.DataFrame({'id': test_id, 'x': test_x.flatten(), 'y': test_predictions.flatten()})
test_output.to_csv('test_predictions.csv', index=False)

# Plotting
plt.scatter(x_sorted, y_sorted, color='red', label='Training Data')
plt.plot(x_sorted, pred_y_train, color='blue', label='Polynomial Fit (Train)')
plt.title('Polynomial Regression Fit')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
