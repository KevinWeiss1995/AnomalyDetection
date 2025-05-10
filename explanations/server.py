from flask import Flask, request, jsonify
from gpt4all import GPT4All
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import logging

app = Flask(__name__)
request_queue = queue.Queue()

def construct_prompt(features, prediction, feature_names, question=None):
    insights = []
    thresholds = {
        'SYN Flag Count': 100,
        'Flow Bytes/s': 10000,
        'Flow IAT Min': 0.0001,
        'RST Flag Count': 100,
        'PSH Flag Count': 1000,
        'Fwd Packets/s': 200,
    }
    
    for feature, threshold in thresholds.items():
        if feature in feature_names:
            idx = feature_names.index(feature)
            value = features[idx]
            if feature == 'SYN Flag Count' and value > threshold:
                insights.append(f"High SYN count: {int(value)} packets")
            elif feature == 'Flow Bytes/s' and value > threshold:
                insights.append(f"High traffic rate: {value:.1f} bytes/s")
            elif feature == 'Flow IAT Min' and value < threshold:
                insights.append(f"Very small packet intervals: {value:.6f}s")

    if not question:
        base_prompt = f"""System: You are a SOC analyst assistant. Analyze the traffic patterns and explain their significance directly to the user. Focus on what makes this traffic suspicious and what risks it poses.

        Traffic Analysis Request:
        Classification: {'Malicious' if prediction > 0.5 else 'Benign'} (confidence: {prediction:.1%})
        Detected Patterns:
        {chr(10).join(f"- {insight}" for insight in insights)}

        Assistant: """
    else:
        base_prompt = f"""System: You are a SOC analyst assistant. Provide specific, actionable advice based on the previous traffic analysis.

        Context:
        {chr(10).join(f"- {insight}" for insight in insights)}

        User: {question}

        Assistant: """

    return base_prompt

class ExplanationServer:
    def __init__(self):
        self.model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
        self.executor = ThreadPoolExecutor(max_workers=20)
        self._load_model()
    
    def _load_model(self):
        self.model.generate("test", max_tokens=1)

server = ExplanationServer()

@app.route('/explain', methods=['POST'])
def explain_traffic():
    data = request.json
    prompt = construct_prompt(
        data['features'],
        data['prediction'],
        data['feature_names'],
        data.get('question')
    )
    
    explanation = server.model.generate(
        prompt,
        max_tokens=150,
        temp=0.7,
        top_k=40,
        top_p=0.4,
        repeat_penalty=1.18
    )
    
    return jsonify({"explanation": explanation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 