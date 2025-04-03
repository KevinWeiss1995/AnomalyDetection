import os
import sys
import tensorflow as tf
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils.git import get_git_repo_root

# Get project root and load model
base_repo = get_git_repo_root()
model_path = os.path.join(base_repo, 'results', 'models', 'network', 'model.keras')
print(f"Looking for model at: {model_path}")  # Debug print

model = tf.keras.models.load_model(model_path)

# Sample 1: Normal Traffic (based on benign patterns in test data)
normal_traffic = {
    "Fwd Packet Length Max": 0.48,  # Most benign samples have lower values
    "Fwd Packet Length Min": 0.0,
    "Bwd Packet Length Min": 0.0,
    "Flow Bytes/s": 0.3,
    "Flow IAT Mean": 0.65,
    "Flow IAT Min": 0.97,
    "Fwd IAT Total": 0.0,
    "Fwd IAT Mean": 0.0,
    "Fwd IAT Min": 0.0,
    "Bwd IAT Total": 0.0,
    "Bwd IAT Mean": 0.0,
    "Bwd IAT Std": 0.0,
    "Bwd IAT Max": 0.0,
    "Bwd IAT Min": 0.0,
    "Fwd PSH Flags": 0.0,
    "Bwd Header Length": 0.43,
    "Fwd Packets/s": 0.35,
    "Bwd Packets/s": 0.48,
    "Min Packet Length": 0.0,
    "Packet Length Std": 0.0,
    "Packet Length Variance": 0.0,
    "FIN Flag Count": 0.0,
    "PSH Flag Count": 0.0,
    "ACK Flag Count": 1.0,
    "URG Flag Count": 1.0,
    "Down/Up Ratio": 0.68,
    "Avg Fwd Segment Size": 0.0,
    "Subflow Fwd Bytes": 0.0,
    "Subflow Bwd Bytes": 0.0,
    "Init_Win_bytes_forward": 0.56,
    "Init_Win_bytes_backward": 0.87,
    "act_data_pkt_fwd": 0.0,
    "Active Mean": 0.0,
    "Active Max": 0.0,
    "Active Min": 0.0,
    "Idle Max": 0.0
}

'''
# Print feature order for debugging
print("\nFeature order used:")
for i, (feature, value) in enumerate(normal_traffic.items()):
    print(f"{i}: {feature} = {value}")

# Load and print training features (if available)
try:
    import pandas as pd
    train_data = pd.read_csv(os.path.join(base_repo, 'data/network/train_data.csv'))
    print("\nTraining features:")
    print(train_data.columns.tolist())
except:
    pass
'''

# Sample 2: Attack Traffic (based on malicious patterns in test data)
attack_traffic = {
    "Fwd Packet Length Max": 0.87,
    "Fwd Packet Length Min": 0.0,
    "Bwd Packet Length Min": 0.0,
    "Flow Bytes/s": 0.32,
    "Flow IAT Mean": 0.84,
    "Flow IAT Min": 0.11,
    "Fwd IAT Total": 0.95,
    "Fwd IAT Mean": 0.83,
    "Fwd IAT Min": 0.38,
    "Bwd IAT Total": 0.85,
    "Bwd IAT Mean": 0.85,
    "Bwd IAT Std": 0.86,
    "Bwd IAT Max": 0.85,
    "Bwd IAT Min": 0.72,
    "Fwd PSH Flags": 0.0,
    "Bwd Header Length": 0.83,
    "Fwd Packets/s": 0.16,
    "Bwd Packets/s": 0.21,
    "Min Packet Length": 0.0,
    "Packet Length Std": 0.83,
    "Packet Length Variance": 0.83,
    "FIN Flag Count": 0.0,
    "PSH Flag Count": 0.0,
    "ACK Flag Count": 1.0,
    "URG Flag Count": 0.0,
    "Down/Up Ratio": 0.0,
    "Avg Fwd Segment Size": 0.97,
    "Subflow Fwd Bytes": 0.96,
    "Subflow Bwd Bytes": 0.86,
    "Init_Win_bytes_forward": 0.39,
    "Init_Win_bytes_backward": 0.77,
    "act_data_pkt_fwd": 0.92,
    "Active Mean": 0.83,
    "Active Max": 0.83,
    "Active Min": 0.83,
    "Idle Max": 0.96
}

# Convert samples to numpy arrays
normal_array = np.array([list(normal_traffic.values())])
attack_array = np.array([list(attack_traffic.values())])

# Make predictions
print("\nAnalyzing Normal Traffic Sample:")
normal_prob = model.predict(normal_array)[0][0]
print(f"Probability of malicious traffic: {normal_prob:.2%}")
print(f"Classification: {'Malicious' if normal_prob > 0.5 else 'Benign'}")

print("\nAnalyzing Attack Traffic Sample:")
attack_prob = model.predict(attack_array)[0][0]
print(f"Probability of malicious traffic: {attack_prob:.2%}")
print(f"Classification: {'Malicious' if attack_prob > 0.5 else 'Benign'}")

# Print suspicious indicators for attack traffic
print("\nSuspicious Indicators in Attack Traffic:")
if attack_traffic["Flow Bytes/s"] > 0.3:
    print(f"- High Flow Rate: {attack_traffic['Flow Bytes/s']:.2f}")
if attack_traffic["Flow IAT Min"] < 0.1:
    print(f"- Very small Inter-Arrival Time: {attack_traffic['Flow IAT Min']}")
if attack_traffic["Packet Length Variance"] > 0.8:
    print(f"- High Packet Length Variance: {attack_traffic['Packet Length Variance']}")
if attack_traffic["act_data_pkt_fwd"] > 0.9:
    print(f"- High Forward Data Packets: {attack_traffic['act_data_pkt_fwd']}")

# Suspicious values in normal traffic:
print("\nPotentially suspicious values in normal traffic:")
suspicious_thresholds = {
    "Flow Bytes/s": 1000,
    "SYN Flag Count": 2,
    "RST Flag Count": 2,
    "Flow IAT Min": 0.0001
}

for feat, threshold in suspicious_thresholds.items():
    if feat in normal_traffic and normal_traffic[feat] > threshold:
        print(f"- High {feat}: {normal_traffic[feat]}") 