"""
Voice Emotion Detection Model Training
Trains a CNN model to classify emotions from audio MFCC features
"""

import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

def extract_mfcc_features(audio_path, n_mfcc=40, max_pad_len=174):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, duration=3, offset=0.5)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
            
        return mfcc
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_dataset(data_path):
    """Load audio dataset and extract features"""
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    X, y = [], []
    
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        if not os.path.exists(emotion_path):
            print(f"Warning: {emotion_path} not found")
            continue
            
        for file in os.listdir(emotion_path):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_path, file)
                features = extract_mfcc_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion)
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    """Create CNN model for emotion classification"""
    model = keras.Sequential([
        keras.layers.Conv1D(64, 5, activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv1D(128, 5, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        
        keras.layers.Conv1D(256, 5, activation='relu'),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_voice_emotion_model(data_path='audio_dataset'):
    """Train the voice emotion detection model"""
    print("Loading dataset...")
    X, y = load_dataset(data_path)
    
    if len(X) == 0:
        print("No data found! Please ensure audio files are in the correct directory structure:")
        print("audio_dataset/")
        print("  ├── neutral/")
        print("  ├── calm/")
        print("  ├── happy/")
        print("  ├── sad/")
        print("  ├── angry/")
        print("  ├── fearful/")
        print("  ├── disgust/")
        print("  └── surprised/")
        return
    
    print(f"Loaded {len(X)} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Reshape for CNN (samples, features, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    model = create_model((X.shape[1], 1), len(label_encoder.classes_))
    
    print("Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model.save('Emotion_Voice_Detection_Model.h5')
    print("Model saved as 'Emotion_Voice_Detection_Model.h5'")
    
    # Save label encoder classes
    np.save('emotion_classes.npy', label_encoder.classes_)
    print("Label classes saved as 'emotion_classes.npy'")
    
    return model, history

if __name__ == "__main__":
    # Train the model
    model, history = train_voice_emotion_model()
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('voice_training_history.png')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting")