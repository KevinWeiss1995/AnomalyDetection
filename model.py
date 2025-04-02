from tensorflow import keras
from tensorflow.keras import layers, Input

def create_model(input_shape):
    """
    Neural net for binary traffic classification with residual connections.
    Uses dropout and batch norm to prevent overfitting.
    
    Args:
        input_shape: Number of input features
    
    Returns:
        A compiled Keras model
    """
    inputs = Input(shape=(input_shape,), name='input')
    
    x = layers.Dropout(0.5)(inputs)
    
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.005),
                    kernel_initializer='he_uniform')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    
    skip = x
    x = layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Add()([x, skip])
    
    x = layers.Dense(16, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model 