import os
import tensorflow as tf
from utils.git import get_git_repo_root
from data.loader import load_data
from training import train_with_cross_validation, train_final_model

def main():
   
    print("Loading data...")
    X, y, X_test, y_test, feature_names = load_data()
    
    fold_scores = train_with_cross_validation(X, y)

    print("\nTraining final model...")
    final_model = train_final_model(X, y, X_test, y_test)

    # Optionally save the model 
    '''
    model_dir = os.path.join(get_git_repo_root(), 'results', 'models', 'network', 'saved_model')
    tf.saved_model.save(final_model, model_dir)
    print(f"\nModel saved to: {model_dir}")
    '''

if __name__ == "__main__":
    main() 