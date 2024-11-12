import numpy as np

# parameters
def initialize_parameters():
    np.random.seed(42)
    weights = {
        'W1': np.random.randn(4, 3) * 0.01,  # 4 hidden units, 3 input features
        'b1': np.zeros((4, 1)),
        'W2': np.random.randn(1, 4) * 0.01,  # 1 output unit, 4 hidden units
        'b2': np.zeros((1, 1))
    }
    return weights


# activation functions
def relu(Z):
    return np.maximum(0, Z)


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# forward propagation
def forward_propagation(X, weights):
    Z1 = np.dot(weights['W1'], X) + weights['b1']
    A1 = relu(Z1)
    Z2 = np.dot(weights['W2'], A1) + weights['b2']
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache


# loss function
def compute_loss(Y, A2):
    m = Y.shape[1]
    loss = -1 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return np.squeeze(loss)


# backward propagation
def backward_propagation(X, Y, cache, weights):
    m = X.shape[1]
    (Z1, A1, Z2, A2) = cache
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(weights['W2'].T, dZ2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return gradients

# parameters update
def update_parameters(weights, gradients, learning_rate):
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b2'] -= learning_rate * gradients['db2']
    return weights

# training
def train(X, Y, weights, epochs=1000, learning_rate=0.01):
    for i in range(epochs):
        A2, cache = forward_propagation(X, weights)
        cost = compute_loss(Y, A2)
        gradients = backward_propagation(X, Y, cache, weights)
        weights = update_parameters(weights, gradients, learning_rate)

        if i % 100 == 0:
            print(f"Cost after epoch {i}: {cost}")
    return weights

X = np.random.randn(3, 100)
Y = (np.sum(X, axis=0) > 0).astype(int).reshape(1, 100)
weights = initialize_parameters()
trained_weights = train(X, Y, weights, epochs=1000, learning_rate=0.05)

def predict(X, weights):
    A2, _ = forward_propagation(X, weights)
    predictions = (A2 > 0.5).astype(int)
    return predictions

# testing on training data
predictions = predict(X, trained_weights)
print("Predictions:", predictions)

# checking accuracy
accuracy = np.mean(predictions == Y) * 100
print(f"Accuracy: {accuracy:.2f}%")

"""
Cost after epoch 0: 0.693146762425115
Cost after epoch 100: 0.6898717406514114
Cost after epoch 200: 0.685374515231065
Cost after epoch 300: 0.6350379741169448
Cost after epoch 400: 0.4111034420111988
Cost after epoch 500: 0.2124382182655117
Cost after epoch 600: 0.14111541583883183
Cost after epoch 700: 0.11051223397826851
Cost after epoch 800: 0.0937210160320114
Cost after epoch 900: 0.08290846043256309
Predictions: [[1 0 1 0 1 1 1 0 0 1 1 1 1 1 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 1 1
  1 0 0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 1 1 0 0 1 1 0 1 1 0 1 1 1
  0 0 0 1 1 1 0 0 1 0 1 0 1 1 1 0 0 0 1 0 1 0 1 1 0 0 1 0]]
Accuracy: 98.00%
"""