import pandas as pd
import numpy as np
import os
from utils.git import get_git_repo_root

def load_data():
    """
    Loads and preprocesses the network traffic data.
    
    Returns:
        X, y: Training data and labels
        X_test, y_test: Test data and labels
        feature_names: List of feature names
    """
    base_repo = get_git_repo_root()
    data_dir = os.path.join(base_repo, 'data', 'network')
  
    train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    train_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(train_data))
    train_data = train_data.iloc[shuffle_idx].reset_index(drop=True)
    train_labels = train_labels.iloc[shuffle_idx].reset_index(drop=True)

    X = train_data.values
    y = train_labels.values.ravel()
    X_test = test_data.values
    y_test = test_labels.values.ravel()

    feature_names = train_data.columns.tolist()
    feature_file_path = os.path.join(base_repo, 'results', 'models', 'network', 'network_features.txt')
    os.makedirs(os.path.dirname(feature_file_path), exist_ok=True)
    
    with open(feature_file_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
            
    return X, y, X_test, y_test, feature_names
