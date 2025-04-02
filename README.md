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
