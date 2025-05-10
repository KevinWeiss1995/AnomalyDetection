import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explanations.client import ExplanationClient, AlertWindow
from PySide6.QtWidgets import QApplication
import sys

def test_alert():
    app = QApplication(sys.argv)
    
    attack_traffic = {
        "SYN Flag Count": 7659,
        "Flow Bytes/s": 12022.69,
        "Flow IAT Min": 0.000007,
        "RST Flag Count": 7659,
        "Fwd Packets/s": 255.80
    }
    
    feature_names = list(attack_traffic.keys())
    features = list(attack_traffic.values())
    prediction = 0.99
    
    client = ExplanationClient()
    alert = AlertWindow(client, features, prediction, feature_names)
    
    explanation = client.get_explanation(features, prediction, feature_names)
    alert.show_alert(explanation)
    
    return app.exec()

if __name__ == "__main__":
    test_alert() 