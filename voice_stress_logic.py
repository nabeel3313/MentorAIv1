# ========== INSTALL REQUIRED PACKAGES ==========
# !pip install resampy librosa numpy tensorflow soundfile -q

# ========== IMPORT EVERYTHING ==========
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
import os
print("✅ All packages imported successfully!")

# ========== LOAD MODEL ==========
def load_emotion_model(model_path='/content/Emotion_Voice_Detection_Model.h5'):
    """Load the emotion detection model"""
    try:
        # Try loading directly
        model = keras.models.load_model(model_path)
        print("✅ Model loaded directly!")
    except Exception as e:
        # If that fails, recreate architecture
        print(f"⚠️ Loading failed: {e}")
        print("Recreating model architecture...")
        model = keras.Sequential([
            keras.layers.Input(shape=(40, 1)),
            keras.layers.Conv1D(128, 5, padding='same'),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.1),
            keras.layers.MaxPooling1D(pool_size=8),
            keras.layers.Conv1D(128, 5, padding='same'),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Flatten(),
            keras.layers.Dense(8),
            keras.layers.Activation('softmax')
        ])
        model.load_weights(model_path)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        print("✅ Model recreated and weights loaded!")

    # Show model info
    print(f"\n📊 Model Info:")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    return model

# Load the model
model = load_emotion_model()

# ========== TEST FUNCTION ==========
def test_emotion_model(audio_path):
    """
    Test the emotion detection model on a single audio file
    """
    print("\n" + "="*60)
    print(f"🎵 Testing: {audio_path}")
    print("="*60)

    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return None, None

        # Step 1: Load audio (simpler method without res_type)
        print("📥 Loading audio file...")
        X, sample_rate = librosa.load(audio_path, sr=None)  # sr=None keeps original rate

        print(f"   Audio length: {len(X)} samples")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Duration: {len(X)/sample_rate:.2f} seconds")

        # Step 2: Extract MFCC features
        print("🔍 Extracting MFCC features...")
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        print(f"   MFCC shape: {mfccs.shape}")
        print(f"   MFCC mean shape: {mfccs_mean.shape}")

        # Step 3: Reshape for model
        mfccs_reshaped = mfccs_mean.reshape(1, 40, 1)

        # Step 4: Predict
        print("🤖 Making prediction...")
        predictions = model.predict(mfccs_reshaped, verbose=0)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions) * 100

        # Step 5: Emotion mapping
        emotion_dict = {
            0: "NEUTRAL",
            1: "CALM",
            2: "HAPPY",
            3: "SAD",
            4: "ANGRY",
            5: "FEARFUL",
            6: "DISGUST",
            7: "SURPRISED"
        }

        # Display results
        print("\n📊 EMOTION PROBABILITIES:")
        print("-"*50)
        emotions = ["Neutral", "Calm", "Happy", "Sad",
                   "Angry", "Fearful", "Disgust", "Surprised"]

        for i, emotion_name in enumerate(emotions):
            prob_percent = predictions[0][i] * 100
            star = " ⭐" if i == predicted_class else ""
            bar = "█" * int(prob_percent / 5)  # Visual bar
            print(f"  {emotion_name:10}: {prob_percent:6.2f}% {bar}{star}")

        print("-"*50)
        print(f"🎯 FINAL PREDICTION: {emotion_dict[predicted_class]}")
        print(f"📈 CONFIDENCE: {confidence:.2f}%")
        print("="*60)

        return emotion_dict[predicted_class], confidence

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# ========== TEST WITH SAMPLE AUDIO ==========
# First, let's create or find a test audio

print("\n" + "="*60)
print("🎤 CREATING TEST AUDIO")
print("="*60)

test_emotion_model('/content/narendra-modi-ji-voice-made-with-Voicemod (1).mp3')