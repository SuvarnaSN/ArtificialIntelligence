# XOR MLP Notebook (AI_HW3)

This repository contains a Jupyter Notebook that builds, trains and evaluates a small Multi-Layer Perceptron (MLP) on a noisy XOR dataset implemented in PyTorch. The notebook generates a synthetic XOR dataset with Gaussian noise, trains a 2-layer MLP with binary cross-entropy (BCE) loss, and visualizes dataset points and training/validation curves.

---

## Notebook overview

File: `AI_HW3 (2).ipynb`

What the notebook does:
- Creates a noisy XOR dataset by sampling around the four canonical XOR points `[[0,0],[0,1],[1,0],[1,1]]` and adding Gaussian noise.
- Splits the generated data into training and validation sets (default test_size=0.2).
- Defines a small MLP class:
  - Input size = 2
  - Hidden size = configurable (example uses 4)
  - Single output neuron for binary classification (BCEWithLogitsLoss used)
  - Activation selectable (`'relu'` or `'tanh'`)
- Trains the model using Adam optimizer and prints per-epoch training/validation loss and accuracy.
- Plots:
  - The noisy XOR dataset with noiseless points highlighted
  - Training and validation loss over epochs
  - Training and validation accuracy over epochs

A run of the notebook (example run included in the cell outputs) achieves very high training and validation accuracy on the noisy XOR dataset.

---

## Main hyperparameters (defaults used in the notebook)

- Gaussian noise standard deviation (`stddeviation`): 0.07
- Epochs (`num_epochs`): 150
- Learning rate (`lr`): 0.01
- Validation split (`test_size`): 0.2
- Hidden units (`num_hidden`): 4
- Loss function: `nn.BCEWithLogitsLoss`
- Optimizer: `torch.optim.Adam`

---

## Key files / code sections

- Data creation: `XORDataset(stddeviation, noiseless_main_points, noiseless_coreOutputLabels)`
  - Generates noisy points and concatenates them with the 4 noiseless XOR points.
- Model: class `MLP(nn.Module)`
  - Two linear layers: input -> hidden -> 1
  - Activation chosen by passing `'relu'` or `'tanh'`
- Training & evaluation function: `calculateAccuracyandLoss(modl, XtrainVal, yTrain, XptVal, yLabelVal)`
  - Trains the model and stores training/validation loss & accuracy in lists.
  - Note: The function currently references `model` in one place during validation (should use `modl`) — see Known Issues below.

---

## How to run (recommended)

1. Install dependencies:
   - Python 3.7+
   - PyTorch (matching your environment; CPU or GPU)
   - numpy
   - matplotlib
   - scikit-learn

   Example (CPU-only):
   ```
   pip install torch torchvision torchaudio numpy matplotlib scikit-learn
   ```

2. Open the notebook in Jupyter or Colab:
   - In Colab: `File -> Upload notebook` and run.
   - In local Jupyter: `jupyter notebook` and open `AI_HW3 (2).ipynb`.

3. Run all cells. Adjust hyperparameters at the top of the notebook as you like.

---

## Interpreting outputs

- The scatter plot shows all noisy points colored by label and the four canonical noiseless XOR points highlighted.
- Console prints per epoch:
  - `Train Loss`, `Validation Loss`
  - `Train Accuracy`, `Validation Accuracy`
- Final printed values show the final training and validation accuracy.
- Final pair of plots:
  - Left: Train/Validation BCE loss vs epoch
  - Right: Train/Validation accuracy vs epoch

---
Thank you — tell me which of the suggested improvements you'd like me to apply and I can prepare an updated notebook or a corrected code cell for you.
