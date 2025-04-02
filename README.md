# Anomaly Detection in Network Traffic

## Overview

This project trains a binary classifier to determine abnormalities in network traffic. A variety of techniques are used to improve performance such as K-Fold cross validation, 
natural gradient descent (KFAC), and a cyclic learning rate. The model acheives ~90% accuracy leaving a small amount of room for improvement. The model is trained with the CIC-IDS 2017 dataset 
which can be found here: https://www.unb.ca/cic/datasets/ids-2017.html

## Preprocessing

The preprocessing script can be found in the preprocessing folder, and handles the following tasks:
- Removes identifying and metadata features (Flow ID, IPs, ports, etc.)
- Drops columns containing all zeros
- Converts labels to binary classification (Normal/Attack)
- Handles class imbalance using RandomUnderSampler
- Removes low-importance features using RandomForest feature importance
- Scales data using QuantileTransformer
- Splits data into train/test sets
- Saves processed datasets and transformer for later use

## Model

The model is a neural network that consists of 13 layers: 

- 5 Dense layers
- 4 Dropout layers
- 2 BatchNorm layers
- 1 Add layer (for residual connection)
- 1 Input layer

The architecture includes a residual block (layers 6-10) that helps with gradient flow during training.

## Training

The model is trained utilizing several techniques:

1. 5-Fold Cross Validation: The dataset is split into 5 parts, where we train on 4 parts and validate on the remaining part, rotating through all combinations. This gives us a robust estimate of model performance and helps detect overfitting.

2. Cyclic Learning Rate (CLR): Instead of using a fixed learning rate, we cycle between 0.0001 and 0.001, which helps escape local minima and often leads to better convergence. The learning rate oscillates over a fixed step size of 2000 batches.

3. KFAC (K-FAC) Optimizer: We use K-FAC (Kronecker-Factored Approximate Curvature) for natural gradient descent, which helps the model converge faster and more reliably than traditional optimizers by approximating the Fisher Information Matrix.

4. Early Stopping: Training automatically stops when validation loss stops improving, preventing overfitting. The best weights are restored.

The final model is trained on the full dataset after validation confirms the architecture's effectiveness.
