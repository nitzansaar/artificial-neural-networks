import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time

# ---------- 1. Load and preprocess data ----------
def load_data(data_dir):
    X, y = [], []
    for label, folder in enumerate(["cat", "dog"]):
        path = os.path.join(data_dir, folder)
        for file in os.listdir(path)[:500]:      # limit to speed up
            img = Image.open(os.path.join(path, file)).convert("RGB").resize((64, 64))
            X.append(np.array(img) / 255.0)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data("data/train")
idx = np.arange(len(X))
np.random.shuffle(idx)
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print("Train set:", X_train.shape, "Test set:", X_test.shape)
# print("Train cats:", (y_train==0).sum(), "dogs:", (y_train==1).sum())
# print("Test cats:", (y_test==0).sum(), "dogs:", (y_test==1).sum())

# ---------- 2. Helper functions ----------
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)
def mse_loss(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
def mse_grad(y_true, y_pred): return 2 * (y_pred - y_true) / y_true.size

# ---------- 3. Convolution & pooling ----------
def conv2d(img, kernel):
    h, w, c = img.shape
    kh, kw, kc = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = img[i:i+kh, j:j+kw, :]
            out[i, j] = np.sum(region * kernel)
    return out

def maxpool(img, size=2):
    h, w = img.shape
    new_h, new_w = h // size, w // size
    pooled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            pooled[i, j] = np.max(img[i*size:(i+1)*size, j*size:(j+1)*size])
    return pooled

# ---------- 4. Initialize parameters ----------
conv_filter = np.random.randn(3, 3, 3) * 0.1
W_fc = np.random.randn(31 * 31, 1) * 0.01
b_fc = np.zeros((1,))

# ---------- 5. Training loop ----------
lr = 0.001
epochs = 5

print("\n--- Training ---")
train_start = time.time()

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_train)):
        img = X_train[i]
        label = np.array([y_train[i]])

        # Forward pass
        conv_out = conv2d(img, conv_filter)
        relu_out = relu(conv_out)
        pooled = maxpool(relu_out)
        flat = pooled.flatten()
        y_pred = relu(np.dot(flat, W_fc) + b_fc)

        # Compute loss
        loss = mse_loss(label, y_pred)
        total_loss += loss

        # Backprop through FC layer only
        grad_y = mse_grad(label, y_pred) * relu_derivative(y_pred)
        grad_W = np.outer(flat, grad_y)
        grad_b = grad_y

        # Update weights
        W_fc -= lr * grad_W
        b_fc -= lr * grad_b

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.4f}")

train_end = time.time()
train_time = train_end - train_start
print(f"Training time: {train_time:.2f} seconds")

# ---------- 6. Evaluation ----------
print("\n--- Evaluation ---")
eval_start = time.time()

correct = 0
for i in range(len(X_test)):
    conv_out = conv2d(X_test[i], conv_filter)
    relu_out = relu(conv_out)
    pooled = maxpool(relu_out)
    flat = pooled.flatten()
    y_pred = relu(np.dot(flat, W_fc) + b_fc)
    prediction = int((y_pred > 0.5).item())
    if prediction == y_test[i]:
        correct += 1

eval_end = time.time()
eval_time = eval_end - eval_start

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Evaluation time: {eval_time:.2f} seconds")

# ---------- 7. Summary ----------
print("\n--- Summary ---")
print(f"Device: CPU (NumPy)")
print(f"Training time: {train_time:.2f} sec | Evaluation time: {eval_time:.2f} sec | Accuracy: {accuracy*100:.2f}%")