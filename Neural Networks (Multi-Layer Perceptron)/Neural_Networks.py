import numpy as np

# Generate a simple linear dataset: y = 2x + 1 (with a bit of noise)

np.random.seed(0)  # for reproducibility
X = np.linspace(0, 1, 20)  # 20 points between 0 and 1
true_w = 2.0
true_b = 1.0

# Generate y = 2x + 1 + noise
noise = np.random.normal(scale=0.1, size=X.shape)  # small Gaussian noise
y = true_w * X + true_b + noise

print("X values:", X)
print("True underlying function: y = 2*x + 1")
print("First 5 generated y values:", y[:5])

# Initialize the weight and bias of the neuron with random values
np.random.seed(42)  # seed for reproducible random results
w = np.random.randn()  # random initial weight
b = np.random.randn()  # random initial bias

print(f"Initial weight (w) = {w:.3f}")
print(f"Initial bias (b) = {b:.3f}")

# Training the single neuron using gradient descent
learning_rate = 0.1   # step size for weight/bias updates
epochs = 100          # how many passes over the data

# Lists to store the loss at each epoch for later visualization (optional)
losses = []

for epoch in range(epochs):
    # 1. Compute predictions for all data points in X
    y_pred = w * X + b

    # 2. Calculate the error (difference) for each point
    error = y_pred - y

    # 3. Compute Mean Squared Error (MSE) loss
    loss = (error ** 2).mean()
    losses.append(loss)

    # 4. Compute gradients:
    #    dLoss/dw and dLoss/db (analytical derivatives for linear model)
    grad_w = 2 * (error * X).mean()    # derivative of MSE w.rt. w
    grad_b = 2 * error.mean()         # derivative of MSE w.rt. b

    # 5. Update parameters using the gradients
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # (Optional) Print loss every 20 epochs to track progress
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, w = {w:.3f}, b = {b:.3f}")

# After training, print the final learned parameters and loss
print(f"Final learned weight (w) = {w:.3f}")
print(f"Final learned bias (b) = {b:.3f}")
print(f"Final loss = {losses[-1]:.6f}")

import matplotlib.pyplot as plt

# Plot the data points
plt.figure(figsize=(6,4))
plt.scatter(X, y, label='Data (true y)', color='blue')

# Plot the initial line (before training)
X_line = np.array([0, 1])  # just need two points (0 and 1) to draw a line
y_initial_line = w * X_line + b  # careful: w and b have been updated, we need initial values
# Actually, let's reuse the initial w and b we saved (if we saved them)
# If not saved, reinitialize for demonstration (in a real run, you'd save them before training)
w_initial = 0.497  # (for reproducibility, this was the initial w printed above)
b_initial = -0.138 # (initial b printed above)
y_initial_line = w_initial * X_line + b_initial
plt.plot(X_line, y_initial_line, 'r--', label='Initial Guess')

# Plot the learned line (after training)
y_final_line = w * X_line + b
plt.plot(X_line, y_final_line, 'g-', label='Learned Line')

# Add labels and legend
plt.title("Single Neuron Fit to y = 2x + 1")
plt.xlabel("x (input)")
plt.ylabel("y (output)")
plt.legend()
plt.show()

import numpy as np

# Generate toy data again (y = 2x + 1 + noise)
np.random.seed(0)
X = np.linspace(0, 1, 20)
true_w, true_b = 2.0, 1.0
noise = np.random.normal(scale=0.1, size=X.shape)
# Now let's generate data with a curved pattern
y = true_w * X + true_b + 0.5 * np.sin(5 * X) + noise

# Initialize weights and biases for ten neurons in the hidden layer
np.random.seed(42)
hidden_size = 10
w1 = np.random.randn(hidden_size)          # shape: (10,)
b1 = np.random.randn(hidden_size)          # shape: (10,)
w2 = np.random.randn(hidden_size)          # shape: (10,) — 1 output neuron receiving from 10 hidden
b2 = np.random.randn()                     # scalar bias for output

# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Training loop
lr = 0.1
epochs = 200
losses = []

for epoch in range(epochs):
    # === Forward pass ===
    # Expand input X to shape (num_samples, 1) so broadcasting works
    X_input = X[:, np.newaxis]                # shape: (20, 1)

    # Forward pass through hidden layer (broadcasting handles shape)
    z1 = X_input * w1 + b1                    # shape: (20, 10)
    a1 = relu(z1)                             # shape: (20, 10)

    # Forward pass through output layer: dot product across hidden neurons
    y_pred = np.dot(a1, w2) + b2              # shape: (20,)

    # === Compute MSE Loss ===
    error = y_pred - y
    loss = (error ** 2).mean()
    losses.append(loss)

    # === Backpropagation ===
    grad_y_pred = 2 * (y_pred - y) / len(X)     # shape: (20,)

    # Output layer gradients
    grad_w2 = np.dot(a1.T, grad_y_pred)         # shape: (10,)
    grad_b2 = np.sum(grad_y_pred)               # scalar

    # Hidden layer gradients
    grad_a1 = grad_y_pred[:, np.newaxis] * w2   # shape: (20,10)
    grad_z1 = grad_a1 * relu_deriv(z1)          # shape: (20,10)

    grad_w1 = np.sum(X_input * grad_z1, axis=0) # shape: (10,)
    grad_b1 = np.sum(grad_z1, axis=0)           # shape: (10,)

    # === Update weights ===
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2

    # Print every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

import matplotlib.pyplot as plt

# Generate dense x values for a smooth curve
X_dense = np.linspace(0, 1, 200)
X_dense_input = X_dense[:, np.newaxis]            # shape: (200, 1)

# Forward pass with vectorized hidden layer
z1_dense = X_dense_input * w1 + b1                # shape: (200, 10)
a1_dense = relu(z1_dense)                         # shape: (200, 10)
y_pred_dense = np.dot(a1_dense, w2) + b2          # shape: (200,)

# Plot the learned curve
plt.figure(figsize=(6, 4))
plt.scatter(X, y, label='True Data', color='blue')
plt.plot(X_dense, y_pred_dense, label='MLP Prediction', color='green')
plt.title("MLP with 10 Hidden Neurons (ReLU)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

from tensorflow import keras

# Load MNIST dataset (handwritten digit images)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# For this exercise, use a subset for faster training (you can increase these numbers later)
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# Print shapes to understand the data structure
print("Training data shape (images):", x_train.shape)  # (number, 28, 28)
print("Training labels shape:", y_train.shape)         # (number,)
print("Test data shape (images):", x_test.shape)
print("Test labels shape:", y_test.shape)

# Normalize pixel values to [0,1] by dividing by 255
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Flatten 28x28 images into 784-dimensional vectors
x_train = x_train.reshape(len(x_train), 28*28)
x_test  = x_test.reshape(len(x_test), 28*28)

print("Training data shape after flattening:", x_train.shape)  # (number, 784)
print("First 5 training labels:", y_train[:5])

from tensorflow.keras import layers

# Build a simple MLP: 784 -> [32 ReLU] -> [10 Softmax]
model = keras.Sequential([
    keras.Input(shape=(784,)),  # Define the input shape using Input layer
    layers.Dense(32, activation='relu'), # Hidden layer with 32 neurons
    layers.Dense(10, activation='softmax') # Output layer with 10 neurons (0-9)
])

# Compile the model with an optimizer, loss, and metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Train the model on the training data
epochs = 5  # number of passes through the training dataset

history = model.fit(x_train, y_train,
                    epochs=epochs,
                    validation_data=(x_test, y_test))

import matplotlib.pyplot as plt

# Extract accuracy history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(acc)+1)

# Plot accuracy curves
plt.figure(figsize=(6,4))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

import numpy as np

# Get model predictions for the test set
pred_probs = model.predict(x_test)              # predicted probabilities for each class (shape: [num_test, 10])
pred_labels = np.argmax(pred_probs, axis=1)     # convert probabilities to class with highest probability

# Select a few random test samples to visualize
num_samples_to_show = 5
indices = np.random.choice(len(x_test), num_samples_to_show, replace=False)

plt.figure(figsize=(10, 2))
for i, idx in enumerate(indices):
    img = x_test[idx].reshape(28, 28)  # reshape back to 28x28 image for display
    true_label = y_test[idx]
    predicted_label = pred_labels[idx]

    # Plot the image
    plt.subplot(1, num_samples_to_show, i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # no axis for cleaner look

    # Title with True and Predicted labels
    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontdict={'fontsize': 10})
plt.show()


