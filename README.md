# Anomaly Detection in Network Traffic

## Overview

This project trains a binary classifier to determine abnormalities in network traffic. A variety of techniques are used to improve performance such as K-Fold cross validation, 
natural gradient descent (via KFAC), and a cyclic learning rate. The model acheives ~90% accuracy leaving a small amount of room for improvement. The model is trained on the CIC-IDS 2017 dataset 
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

The architecture includes a residual block (layers 6-10). Residual connections were originally used to help with gradient flow during training, but our KFAC optimizer handles this well. We still implement them because they create ensemble-like behavior where the network can choose to use or bypass certain layers and provide additional paths for information flow, regardless of optimizer.

## Training

### KFAC (Kronecker-Factored Approximate Curvature)
KFAC is an optimization technique that approximates natural gradient descent. While traditional optimizers like Adam or SGD only use first-order information (gradients), KFAC uses second-order information (curvature) to make better update steps:

- Traditional optimizers might zigzag in steep valleys of the loss landscape
- KFAC approximates the Fisher Information Matrix using Kronecker products
- This gives better estimates of the optimal step direction
- Result: Faster convergence and better training stability

The tradeoff is increased computational cost per step, but this is often offset by needing fewer steps overall.

### Other Training Techniques
1. 5-Fold Cross Validation: The dataset is split into 5 parts, where we train on 4 parts and validate on the remaining part, rotating through all combinations. This gives us a robust estimate of model performance and helps detect overfitting.

2. Cyclic Learning Rate (CLR): Instead of using a fixed learning rate, we cycle between 0.0001 and 0.001, which helps escape local minima and often leads to better convergence. The learning rate oscillates over a fixed step size of 2000 batches.

3. Early Stopping: Training automatically stops when validation loss stops improving, preventing overfitting. The best weights are restored.

The final model is trained on the full dataset after validation confirms the architecture's effectiveness.


## Results

The model achieves strong performance:

- Overall Accuracy: 90%
- Normal Traffic (Class 0):
  - Precision: 88%
  - Recall: 94%
  - F1-Score: 91%
- Attack Traffic (Class 1):
  - Precision: 92%
  - Recall: 85%
  - F1-Score: 89%

The confusion matrix showed:
- True Negatives (Normal correctly identified): 155,706
- False Positives (Normal misclassified as Attack): 9,939
- False Negatives (Attack misclassified as Normal): 20,875
- True Positives (Attack correctly identified): 119,889

These results indicate the model is well-balanced between classes, with slightly better performance at identifying normal traffic (94% recall) than attacks (85% recall). The high precision for attack detection (92%) means we have relatively few false alarms.

## Testing

The repo includes a script for testing the model on a sample of data. You can generate a sample of data by running traffic_generator.py and traffic_modifier.py simultaneously. Below is an example of the output of use_model.py:

```
Analyzing Normal Traffic Sample:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step
Probability of malicious traffic: 34.81%
Classification: Benign

Analyzing Attack Traffic Sample:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step
Probability of malicious traffic: 82.57%
Classification: Malicious

Suspicious Indicators in Attack Traffic:
- High Flow Rate: 0.32
- High Packet Length Variance: 0.83
- High Forward Data Packets: 0.92

Potentially suspicious values in normal traffic:
- High Flow IAT Min: 0.97
```