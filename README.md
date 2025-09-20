# Needle: A Deep Learning Framework

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CMake](https://img.shields.io/badge/CMake-3.2+-red.svg)](https://cmake.org/)

**Needle** (Necessary Elements of Deep Learning) is a comprehensive deep learning framework built from scratch as part of Carnegie Mellon University's 10-714 Deep Learning Systems course. This educational project implements core components of modern deep learning frameworks including automatic differentiation, neural network modules, and multi-backend support.

## 🚀 Features

### Core Components
- **Automatic Differentiation**: Complete implementation of reverse-mode automatic differentiation with computational graph support
- **Tensor Operations**: Comprehensive set of tensor operations including arithmetic, linear algebra, and element-wise operations
- **Neural Network Modules**: Implementation of common neural network layers (Linear, Conv2d, BatchNorm, ReLU, etc.)
- **Optimizers**: Various optimization algorithms (SGD, Adam, etc.)
- **Data Loading**: Utilities for loading and preprocessing datasets (CIFAR-10, Penn Treebank)

### Multi-Backend Support
- **CPU Backend**: High-performance CPU implementation using C++ with pybind11
- **CUDA Backend**: GPU acceleration for NVIDIA GPUs
- **Metal Backend**: Apple Silicon GPU acceleration using Metal Performance Shaders
- **NumPy Backend**: Pure Python implementation for development and testing

### Advanced Features
- **Convolutional Neural Networks**: 2D convolution operations with padding and stride support
- **Recurrent Neural Networks**: LSTM and RNN implementations for sequence modeling
- **Memory Management**: Efficient memory allocation and deallocation
- **Gradient Checking**: Built-in numerical gradient verification

## 📁 Project Structure

```
needle/
├── python/needle/           # Core Python implementation
│   ├── autograd.py         # Automatic differentiation engine
│   ├── ops.py              # Tensor operations
│   ├── nn.py               # Neural network modules
│   ├── optim.py            # Optimization algorithms
│   ├── data.py             # Data loading utilities
│   └── backend_ndarray/    # Compiled backend modules
├── src/                    # C++/CUDA/Metal source code
│   ├── ndarray_backend_cpu.cc
│   ├── ndarray_backend_cuda.cu
│   └── ndarray_backend_metal.cpp
├── apps/                   # Example applications
│   ├── simple_training.py  # CIFAR-10 training example
│   ├── models.py           # Model definitions
│   └── mlp_resnet.py       # ResNet implementation
├── tests/                  # Comprehensive test suite
├── data/                   # Dataset storage
└── hw*.ipynb              # Course homework notebooks
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CMake 3.2+
- C++ compiler (GCC/Clang)
- CUDA toolkit (optional, for GPU support)
- Xcode command line tools (macOS, for Metal support)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd needle
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the framework:**
   ```bash
   make
   ```

   This will compile the C++/CUDA/Metal backends and create the necessary Python extensions.

### Backend-Specific Setup

#### CUDA Backend (Optional)
- Install CUDA toolkit
- Ensure `nvidia-smi` is available
- The build system will automatically detect and compile CUDA support

#### Metal Backend (macOS)
- Requires macOS with Apple Silicon
- Xcode command line tools must be installed
- Metal support is automatically enabled on macOS

## 🚀 Quick Start

### Basic Tensor Operations

```python
import needle as ndl
import needle.nn as nn

# Create tensors
x = ndl.Tensor([1, 2, 3, 4], dtype="float32")
y = ndl.Tensor([5, 6, 7, 8], dtype="float32")

# Basic operations
z = x + y
print(z)  # [6, 8, 10, 12]

# Automatic differentiation
z = (x * y).sum()
z.backward()
print(x.grad)  # [5, 6, 7, 8]
```

### Neural Network Example

```python
import needle as nn
import needle.optim as optim

# Define a simple MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

# Create model and optimizer
model = MLP(784, 128, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.SoftmaxLoss()

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        
        loss.backward()
        optimizer.step()
        optimizer.reset_grad()
```

### CIFAR-10 Training

```python
# Run the example training script
python apps/simple_training.py
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_ndarray.py      # Tensor operations
python -m pytest tests/test_nn_and_optim.py # Neural networks
python -m pytest tests/test_conv.py         # Convolution operations
```

## 📚 Course Context

This framework is developed as part of CMU's 10-714 Deep Learning Systems course, covering:

- **Homework 1**: Automatic differentiation and basic tensor operations
- **Homework 2**: Neural network modules and optimizers
- **Homework 3**: Backend implementations (CPU/CUDA/Metal)
- **Homework 4**: Convolutional and recurrent neural networks

Each homework builds upon the previous implementations, creating a complete deep learning framework.

## 🔧 Backend Details

### CPU Backend
- Implemented in C++ with pybind11 bindings
- Optimized for performance with SIMD instructions
- Supports all tensor operations

### CUDA Backend
- GPU acceleration for NVIDIA hardware
- Custom CUDA kernels for key operations
- Automatic memory management

### Metal Backend
- Apple Silicon GPU acceleration
- Metal Performance Shaders integration
- Optimized for M1/M2 Macs

## 📖 Documentation

- **API Reference**: See docstrings in individual modules
- **Examples**: Check `apps/` directory for usage examples
- **Tests**: Comprehensive test suite in `tests/` directory

## 🤝 Contributing

This is an educational project. For questions or issues related to the course, please refer to the course materials and discussion forums.

## 📄 License

This project is part of Carnegie Mellon University's 10-714 Deep Learning Systems course. Please refer to the course guidelines for usage and distribution.

## 🙏 Acknowledgments

- Carnegie Mellon University 10-714 course staff
- PyTorch team for inspiration on API design
- The open-source community for various tools and libraries

---

**Note**: This is an educational implementation and should not be used for production workloads. For production deep learning, consider established frameworks like PyTorch or TensorFlow.