# **Bgrad: A Tiny Autograd + Neural Network Engine (Inspired by Andrej Karpathy)**

**Bgrad** is a minimal, educational neural-network and autograd engine implemented from scratch in Python.
It is inspired by Andrej Karpathy’s micrograd, rebuilt step-by-step to help understand:

* how computation graphs work
* how gradients flow backward
* how parameters update during training
* how a multilayer perceptron (MLP) learns from data

This project is intentionally simple and fully transparent. Every computation — forward pass, backward pass, gradient accumulation, and weight update — is written in Python using only basic operations.

---

## **Features**

* Tiny autograd engine with `Value` objects
* Fully manual computation graph construction
* Backpropagation through arbitrary expressions
* `Neuron`, `Layer`, and `MLP` classes
* Easy parameter inspection and gradient descent
* Perfect for learning how neural networks work internally

---

## **Project Structure**

```
Bgrad/
│
├── core/
│   ├── __init__.py
│   ├── value.py
│   ├── nn_module.py     # Neuron, Layer, MLP
│
├── notebooks/
│   └── experiments.ipynb  # optional Jupyter work
│
└── README.md
```

---

## **Installation / Setup**

Add the project directory to your Python path:

```python
import sys
from pathlib import Path

project_root = Path("/Users/dheerajkumar/Developer/AI/Bgrad")  # adjust this path
sys.path.append(str(project_root))
```

Then import components:

```python
from core import MLP, Layer, Neuron
```

---

## **Example: Training a Tiny Neural Network**

```python
from core import MLP

# Inputs
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

# Outputs for each set of inputs
yexpected = [1.0, -1.0, -1.0, 1.0]

# Create network with 3 inputs and layers [4, 4, 1]
nn = MLP(3, [4, 4, 1])

# Train using manual gradient descent
iterations = 200

for k in range(iterations):
    # Forward pass
    ypredicted = [nn(x) for x in xs]
    loss = sum((ypred - yexp)**2 for ypred, yexp in zip(ypredicted, yexpected))

    # Backward pass: compute gradients
    loss.backward()

    # Gradient descent step
    learning_rate = 0.1
    for p in nn.parameters():
        p.data -= learning_rate * p.grad

    # Reset gradients
    nn.zero_grad()

    print(k, loss.data)

# Final predictions
ypredicted = [nn(x) for x in xs]
ypredicted
```

---

## **Why This Project Exists**

This repository is meant for anyone who wants to understand neural networks at the lowest level — without relying on large frameworks like PyTorch or TensorFlow.

By building everything manually, it becomes clear:

* how gradients are computed
* why backpropagation works
* how parameter updates minimize loss
* how layers communicate in an MLP
* what autodiff systems actually do under the hood

This project is an educational foundation for deeper machine-learning exploration.

---

## **Acknowledgment**

Inspired heavily by **Andrej Karpathy’s micrograd**, with modifications and experimentation added during the learning process.