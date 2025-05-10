from tensorflow import keras
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

'''
A simple model with two hidden layers and a sigmoid output layer. 
The model uses dropout and L2 regularization to prevent overfitting.
Binary classification problem.
'''

def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_shape,), kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Add prediction with explanation method
    def predict_with_explanation(self, X, feature_names=None):
        from explanations.llm_explainer import NetworkExplainer
        
        predictions = self.predict(X)
        explainer = NetworkExplainer()
        
        results = []
        for i, (features, pred) in enumerate(zip(X, predictions)):
            explanation = explainer.explain_prediction(features, pred, feature_names)
            results.append({
                'prediction': bool(pred > 0.5),
                'confidence': float(pred),
                'explanation': explanation
            })
        return results
    
    # Add the method to the model
    model.predict_with_explanation = predict_with_explanation.__get__(model)
    return model