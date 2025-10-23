# MLP Trainer for MNIST / FashionMNIST

I created this README to document the training script you provided. Below you'll find what the script does, how to run it, required dependencies, example interactive inputs, expected outputs, known issues found in the code, and suggested fixes/improvements. Read through the "Known issues & fixes" section before running the script — I describe two real bugs that will prevent the code from running correctly as-is and give the exact fixes.

---

## What this script does

- Loads either MNIST (from a local `mnist.pkl` pickle file) or FashionMNIST (downloaded via torchvision).
- Prepares train/validation/test splits and DataLoaders.
- Builds a configurable fully-connected MLP (user-specified number of hidden layers, neurons, and per-layer activations).
- Trains the model for a number of epochs, printing training & validation accuracy and average training loss each epoch.
- Saves the best model (by validation accuracy) to `model.dat`.
- After training, loads the best model and computes test accuracy and a per-class confusion matrix (saved as PNG).
- Saves plots:
  - example input image: `example.png`
  - confusion matrix: `Confusion_Matrix_MNIST.png` or `Confusion_Matrix_FASHION_MNIST.png`
  - train & validation accuracy over epochs: `TrainAndValidationAccuracyPlot.png`

---

## Dependencies

- Python 3.7+
- numpy
- matplotlib
- torch (PyTorch) and torchvision
- pickle (standard library)

Install with pip (example for CPU PyTorch):
```bash
pip install numpy matplotlib torch torchvision
```

You must also have the `mnist.pkl` file in the working directory if you choose the MNIST option.

---

## How to run

This script is interactive (uses input() prompts). Run it in a terminal or interactive environment:

```bash
python your_script.py
```

Example interactive session:
- When asked: `Opt for one which dataset do you want MNIST, FashionMNIST:` enter `MNIST` or `FashionMNIST`.
  - If you pick `MNIST`, ensure `mnist.pkl` is available.
- When asked: `Enter the no of hidden layers:` enter an integer, e.g., `2`.
- For each layer, enter number of neurons and activation name:
  - Example: `Enter no of neurons in the hidden layer 1 : 256`
  - Example: `Enter activation you want to apply for hidden layer 1: relu`
  - Supported activation names in the code: `sigmoid`, `relu`, `tanh`, `leakyrelu` (case-sensitive)
- Choose optimizer (enter one of the exact strings): `SGD`, `Adam`, `AdaGrad`.
  - Note: learning rates are hard-coded: SGD -> lr = 1e-2, Adam -> lr = 1e-3, AdaGrad -> lr = 1e-2

Outputs (saved in the current working directory):
- `example.png` — visualization of the first training image (MNIST branch)
- `model.dat` — saved model state (best validation accuracy)
- `Confusion_Matrix_MNIST.png` or `Confusion_Matrix_FASHION_MNIST.png`
- `TrainAndValidationAccuracyPlot.png`

---

## Important input conventions & examples

- Activation names must match the strings used in the script (`relu`, `tanh`, `sigmoid`, `leakyrelu`).
- Example minimal run:
  - Dataset: `FashionMNIST`
  - Hidden layers: `2`
  - Layer 1 neurons: `256`, activation: `relu`
  - Layer 2 neurons: `128`, activation: `relu`
  - Optimizer: `Adam`

---

## Where files are written

All artifacts are saved to the current working directory:
- Model: `model.dat`
- Plots: `example.png`, `Confusion_Matrix_*.png`, `TrainAndValidationAccuracyPlot.png`