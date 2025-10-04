"""
Neural Network classifier using TensorFlow/Keras for deep learning approach.
"""

import numpy as np
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from models.base_classifier import BaseClassifier


class NeuralNetworkClassifier(BaseClassifier):
    """Neural Network classifier for exoplanet identification."""
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (128, 64, 32),
                 dropout_rate: float = 0.3, learning_rate: float = 0.001,
                 activation: str = 'relu', random_state: int = 42):
        """
        Initialize Neural Network classifier.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            activation: Activation function for hidden layers
            random_state: Random seed for reproducibility
        """
        super().__init__(model_name="NeuralNetwork")
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.activation = activation
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.history = None
    
    def _build_model(self, input_dim: int, n_classes: int) -> keras.Model:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name=self.model_name)
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(units, activation=self.activation, name=f'hidden_{i+1}'))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        if n_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        else:
            # Multi-class classification
            model.add(layers.Dense(n_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, epochs: int = 50,
            batch_size: int = 32, verbose: int = 1, **kwargs) -> None:
        """
        Train the Neural Network classifier.
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            **kwargs: Additional training parameters
        """
        print(f"Training {self.model_name} with architecture: {self.hidden_layers}...")
        
        # Store feature names if provided
        if 'feature_names' in kwargs:
            self.feature_names = kwargs['feature_names']
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Determine number of classes
        n_classes = len(np.unique(y))
        
        # Build model
        self.model = self._build_model(input_dim=X.shape[1], n_classes=n_classes)
        
        # Setup callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=0
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            validation_data = (X_val, y_val)
        
        # Train the model
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Store training metadata
        self.set_training_metadata({
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs_trained': len(self.history.history['loss']),
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'n_classes': n_classes,
            'final_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]) if validation_data else None
        })
        
        print(f"{self.model_name} training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
        predictions = self.model.predict(X, verbose=0)
        
        # Convert probabilities to class labels
        if predictions.shape[1] == 1:
            # Binary classification
            return (predictions > 0.5).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
        predictions = self.model.predict(X, verbose=0)
        
        # For binary classification, return probabilities for both classes
        if predictions.shape[1] == 1:
            prob_class_1 = predictions.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        
        return predictions
    
    def get_training_history(self) -> Optional[dict]:
        """
        Get training history if available.
        
        Returns:
            Dictionary containing training history
        """
        if self.history is None:
            return None
        
        return self.history.history
