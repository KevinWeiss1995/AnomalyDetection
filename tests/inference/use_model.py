import sys
import os
import pandas as pd
import json


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from utils.git import get_git_repo_root 
from explanations.llm_explainer import NetworkExplainer

project_root = get_git_repo_root()

model_path = os.path.join(project_root, 'results', 'models', 'network', 'model.keras')
print(f"Looking for model at: {model_path}")  # Debug print
model = keras.models.load_model(model_path)

normal_traffic = {
    "Fwd Packet Length Max": 1280.0,
    "Fwd Packet Length Min": 40.0,
    "Bwd Packet Length Min": 40.0,
    "Flow Bytes/s": 800.0,
    "Flow IAT Mean": 0.05,
    "Flow IAT Min": 0.001,
    "Fwd IAT Total": 10.0,
    "Fwd IAT Mean": 0.05,
    "Fwd IAT Min": 0.001,
    "Bwd IAT Total": 10.0,
    "Bwd IAT Mean": 0.05,
    "Bwd IAT Std": 0.01,
    "Bwd IAT Max": 0.1,
    "Bwd IAT Min": 0.001,
    "Fwd PSH Flags": 1,
    "Bwd Header Length": 320,
    "Fwd Packets/s": 20.0,
    "Bwd Packets/s": 15.0,
    "Min Packet Length": 40,
    "Max Packet Length": 1280,
    "Packet Length Mean": 500.0,
    "Packet Length Std": 300.0,
    "Packet Length Variance": 90000.0,
    "FIN Flag Count": 1,
    "SYN Flag Count": 1,
    "RST Flag Count": 0,
    "PSH Flag Count": 1,
    "ACK Flag Count": 10,
    "URG Flag Count": 0,
    "CWE Flag Count": 0,
    "ECE Flag Count": 0,
    "Down/Up Ratio": 0.75,
    "Average Packet Size": 500.0,
    "Avg Fwd Segment Size": 600.0,
    "Avg Bwd Segment Size": 400.0,
    "Fwd Header Length": 320
}

attack_traffic = {
    "Fwd Packet Length Max": 50.0,
    "Fwd Packet Length Min": 44.0,
    "Bwd Packet Length Min": 0.0,
    "Flow Bytes/s": 12022.688875459973,
    "Flow IAT Mean": 0.0039095304696718835,
    "Flow IAT Min": 7.867813110351562e-06,
    "Fwd IAT Total": 59.88227820396423,
    "Fwd IAT Mean": 0.0039095304696718835,
    "Fwd IAT Min": 7.867813110351562e-06,
    "Bwd IAT Total": 0.0,
    "Bwd IAT Mean": 0.0,
    "Bwd IAT Std": 0.0,
    "Bwd IAT Max": 0.0,
    "Bwd IAT Min": 0.0,
    "Fwd PSH Flags": 0,
    "Bwd Header Length": 0,
    "Fwd Packets/s": 255.80189096723348,
    "Bwd Packets/s": 0.0,
    "Min Packet Length": 44,
    "Max Packet Length": 50,
    "Packet Length Mean": 47.0,
    "Packet Length Std": 3.0,
    "Packet Length Variance": 9.0,
    "FIN Flag Count": 0,
    "SYN Flag Count": 7659,
    "RST Flag Count": 7659,
    "PSH Flag Count": 0,
    "ACK Flag Count": 7659,
    "URG Flag Count": 0,
    "CWE Flag Count": 0,
    "ECE Flag Count": 0,
    "Down/Up Ratio": 0.0,
    "Average Packet Size": 47.0,
    "Avg Fwd Segment Size": 47.0,
    "Avg Bwd Segment Size": 0.0,
    "Fwd Header Length": 61272
}

normal_array = np.array([list(normal_traffic.values())])
attack_array = np.array([list(attack_traffic.values())])

print("\nAnalyzing Normal Traffic Sample:")
normal_prob = model.predict(normal_array)[0][0]
print(f"Probability of malicious traffic: {normal_prob:.2%}")
print(f"Classification: {'Malicious' if normal_prob > 0.5 else 'Benign'}")

print("\nAnalyzing Attack Traffic Sample:")
attack_prob = model.predict(attack_array)[0][0]
print(f"Probability of malicious traffic: {attack_prob:.2%}")
print(f"Classification: {'Malicious' if attack_prob > 0.5 else 'Benign'}")

print("\nDetailed Analysis:")
explainer = NetworkExplainer()

if attack_prob > 0.5:
    explanation = explainer.explain_prediction(
        list(attack_traffic.values()),
        attack_prob,
        list(attack_traffic.keys())
    )
    print("\nAnalysis:", explanation)

    while True:
        question = input("\nAsk a question about this malicious traffic (or 'quit' to exit): ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        answer = explainer.ask_followup(
            list(attack_traffic.values()),
            attack_prob,
            list(attack_traffic.keys()),
            question
        )
        print("\nAnswer:", answer)

data_path = os.path.join(project_root, 'data', 'network')
X_test = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'test_labels.csv'))

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=1)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test AUC: {test_auc}")


predictions = (model.predict(X_test) > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

'''
with open('final_history.json', 'r') as f:
    history = json.load(f)

plt.plot(history['accuracy'], label='train_accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot loss
plt.plot(history['loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show() 
'''