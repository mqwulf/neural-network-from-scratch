## About The Project

Developed a fully functional feedforward neural network from scratch using only NumPy, implementing all core components—parameter initialization, forward propagation, loss calculation, backpropagation, and parameter updates without using any machine learning libraries. This project demonstrates a deep understanding of the fundamental mathematics behind neural networks, as well as the ability to implement a machine learning model using foundational programming techniques.


## Key Features

Architecture: Implemented a 2-layer neural network with a hidden layer using ReLU activation and an output layer using sigmoid activation for binary classification tasks.
Forward Propagation: Created a forward propagation function to compute intermediate and final layer activations using matrix multiplications and non-linear activations (ReLU and Sigmoid).
Backpropagation: Implemented backpropagation to compute gradients for each layer, optimizing the model by applying gradient descent.
Loss Calculation: Used binary cross-entropy as the loss function, providing feedback for model optimization and measuring performance over epochs.
Training Process: Designed a training loop to iteratively adjust weights and biases over multiple epochs, with a learning rate hyperparameter, achieving a reduction in loss with each epoch.
Accuracy Assessment: Developed a prediction function to evaluate model accuracy on test data, assessing the model's ability to distinguish between binary classes.


## More Information

Programming Language: Python
Libraries: NumPy for matrix operations
Key Skills: Neural network architecture, backpropagation, gradient descent, matrix operations, binary classification, cross-entropy loss.
Project Outcome: Achieved a model capable of accurate binary classification by effectively learning from synthetic data. The model’s performance was validated by a consistent decrease in training loss and accurate predictions on test data, highlighting proficiency in building neural networks from first principles.

### Result

```
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
```
