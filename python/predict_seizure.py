import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environment
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import joblib
import mne
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GRU, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class SeizurePredictor:
    def __init__(self, output_dir="EEG_Data_new"):
        self.output_dir = output_dir
        self.model = None
        self.scaler = None
       
        model_path = os.path.join(output_dir, 'cnn_gru_seizure_model.h5')
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Loading existing CNN-GRU model from {model_path}")
            try:
                self.model = load_model(model_path)
                self.scaler = joblib.load(scaler_path)
                print("Model and scaler loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
                self.scaler = None
        else:
            print("No existing CNN-GRU model found. Will create a new one.")
            self.scaler = StandardScaler()
    
    def extract_temporal_lobe_data(self, file_path):
        try:
            print(f"Reading EDF file: {file_path}")
            raw = mne.io.read_raw_edf(file_path, preload=True)
            print(f"Available channels: {raw.ch_names}")
            
            # Define channel selection patterns
            temporal_channel_patterns = ['T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T3-', 'T4-', 'T5-', 'T6-', 'T7-', 'T8-', 'temp', 'temporal']
            temporal_channels = [ch for pattern in temporal_channel_patterns for ch in raw.ch_names if pattern.lower() in ch.lower()]
            temporal_channels = list(dict.fromkeys(temporal_channels))
            
            # Fallback to brain signal channels if no temporal channels found
            if not temporal_channels:
                brain_signal_patterns = ['eeg', 'ekg', 'ecg', 'emg', 'channel', 'ch', 'signal', 'fp', 'f', 'c', 'p', 'o']
                temporal_channels = [ch for ch in raw.ch_names if any(pattern in ch.lower() for pattern in brain_signal_patterns)]
            
            # Use first available channel as last resort
            if not temporal_channels and len(raw.ch_names) > 0:
                temporal_channels = [raw.ch_names[0]]
            
            if temporal_channels:
                channel = temporal_channels[0]
                print(f"Using channel: {channel}")
                signals = raw.get_data(picks=channel)
                
                # Check for NaN or infinite values
                if np.isnan(signals).any() or np.isinf(signals).any():
                    print("Warning: Signal contains NaN or infinite values. Replacing with zeros.")
                    signals = np.nan_to_num(signals)
                
                fs = raw.info['sfreq']
                return signals, fs
            else:
                print(f"No suitable channels found in {file_path}")
                return None, None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None, None
    
    def extract_features(self, signals, fs, window_size=4, overlap=0.5):
        if signals is None or fs is None:
            return None, None
        
        print(f"Extracting features from signal with shape {signals.shape} and sampling rate {fs} Hz")
        window_length = int(window_size * fs)
        step = int(window_length * (1 - overlap))
        
        # Check if signal is long enough
        if signals.shape[1] <= window_length:
            print(f"Error: Signal length ({signals.shape[1]}) is shorter than window length ({window_length})")
            return None, None
        
        sequences = []
        labels = []
        
        for i in range(0, signals.shape[1] - window_length, step):
            try:
                window = signals[:, i:i+window_length]
                
                # Check for NaN or infinite values
                if np.isnan(window).any() or np.isinf(window).any():
                    window = np.nan_to_num(window)
                
                # Resample to 125 samples for CNN-GRU model
                resampled_window = np.interp(
                    np.linspace(0, 1, 125),
                    np.linspace(0, 1, window.shape[1]),
                    window[0]
                )
                
                # Reshape for CNN-GRU: (time_steps, features)
                sequence = resampled_window[:, np.newaxis]
                sequences.append(sequence)
                
                # Calculate line length as proxy for abnormal activity
                line_length = np.sum(np.abs(np.diff(window[0])))
                labels.append(line_length)
            except Exception as e:
                print(f"Error processing window at position {i}: {e}")
                continue
        
        if len(sequences) == 0:
            print("No valid sequences were extracted")
            return None, None
        
        X = np.array(sequences)
        y = np.array(labels)
        
        # Remove any remaining NaN or inf values
        if np.isnan(X).any() or np.isinf(X).any():
            print("Warning: Features contain NaN or infinite values. Replacing with zeros.")
            X = np.nan_to_num(X)
        
        if np.isnan(y).any() or np.isinf(y).any():
            print("Warning: Labels contain NaN or infinite values. Replacing with zeros.")
            y = np.nan_to_num(y)
        
        print(f"Extracted {len(X)} feature sequences")
        return X, y
    
    def build_cnn_gru_model(self):
        """Build CNN-GRU model architecture"""
        tf.keras.backend.clear_session()
        model = Sequential()
        
        # CNN layers for feature extraction
        model.add(Conv1D(filters=16, kernel_size=3, strides=1, padding='same', input_shape=(125, 1)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling1D(pool_size=2, strides=2))
        
        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(0.3))
        
        # GRU layers for temporal sequence modeling
        model.add(GRU(64, return_sequences=True))
        model.add(GRU(32))
        model.add(Dropout(0.3))
        
        # Dense layers for classification
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, edf_file):
        file_name = os.path.basename(edf_file).split('.')[0]
        output_img = os.path.join(self.output_dir, f"{file_name}_prediction.png")
        
        print(f"Starting prediction for file: {edf_file}")
        signals, fs = self.extract_temporal_lobe_data(edf_file)
        
        if signals is None or fs is None:
            print(f"Could not extract signal from {edf_file}")
            return {
                "file": edf_file,
                "prediction": "ERROR",
                "confidence": "0%",
                "visualization": None
            }
        
        X, y = self.extract_features(signals, fs)
        
        if X is None or len(X) == 0:
            print(f"Could not extract features from {edf_file}")
            return {
                "file": edf_file,
                "prediction": "ERROR",
                "confidence": "0%",
                "visualization": None
            }
        
        # Create model if it doesn't exist
        if self.model is None:
            print("Creating a CNN-GRU model for demonstration purposes")
            
            # Create binary pseudo-labels based on line length
            try:
                threshold = np.percentile(y, 85)  # Top 15% are considered abnormal
                y_binary = (y > threshold).astype(int)
                
                # Scale features
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.fit_transform(X_reshaped)
                X_processed = X_scaled.reshape(X.shape)
                
                # Build and train model
                self.model = self.build_cnn_gru_model()
                
                # Simple training with early stopping
                callbacks = [
                    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                ]
                
                print("Training CNN-GRU model...")
                self.model.fit(
                    X_processed, y_binary,
                    epochs=20,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Save model and scaler
                self.model.save(os.path.join(self.output_dir, 'cnn_gru_seizure_model.h5'))
                joblib.dump(self.scaler, os.path.join(self.output_dir, 'feature_scaler.pkl'))
                print("Model created and saved")
            except Exception as e:
                print(f"Error creating model: {e}")
                return {
                    "file": edf_file,
                    "prediction": "ERROR",
                    "confidence": "0%",
                    "visualization": None
                }
        
        # Make predictions
        try:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.transform(X_reshaped)
            X_processed = X_scaled.reshape(X.shape)
            
            predictions_prob = self.model.predict(X_processed, verbose=0)
            predictions = (predictions_prob > 0.5).astype(int)
            
            # Calculate seizure metrics
            seizure_ratio = np.mean(predictions)
            max_prob = np.max(predictions_prob)
            has_seizure = seizure_ratio > 0.15 or max_prob > 0.85
            
            # Calculate confidence
            confidence = max(seizure_ratio, max_prob) if has_seizure else 1.0 - max_prob
            confidence = max(0.5, confidence)  # Minimum 50% confidence
            
            # Create prediction visualization
            self._create_prediction_visualization(signals, fs, predictions_prob.flatten(), predictions.flatten(), edf_file, output_img)
            
            return {
                "file": edf_file,
                "prediction": "SEIZURE" if has_seizure else "NO SEIZURE",
                "confidence": f"{confidence:.2%}",
                "visualization": output_img
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                "file": edf_file,
                "prediction": "ERROR",
                "confidence": "0%",
                "visualization": None
            }
    
    def _create_prediction_visualization(self, signals, fs, probabilities, predictions, edf_file, output_file):
        print(f"Creating visualization and saving to {output_file}")
        plt.figure(figsize=(12, 10))
        
        # EEG Signal plot
        plt.subplot(4, 1, 1)
        time = np.arange(signals.shape[1]) / fs
        plt.plot(time, signals[0], 'b-', linewidth=1)
        plt.title('EEG Signal (Selected Channel)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Power Spectral Density
        plt.subplot(4, 1, 2)
        chunk_length = min(int(fs * 5), signals.shape[1])
        freqs, psd = welch(signals[0, :chunk_length], fs, nperseg=min(1024, chunk_length))
        plt.semilogy(freqs, psd)
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.xlim([0, min(100, fs/2)])
        
        # Spectrogram
        plt.subplot(4, 1, 3)
        plt.specgram(signals[0], NFFT=256, Fs=fs, noverlap=128, cmap='viridis')
        plt.title('Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar(format='%+2.0f dB')
        
        # Seizure Probability
        plt.subplot(4, 1, 4)
        window_size = 4
        overlap = 0.5
        window_step = window_size * (1 - overlap)
        window_times = np.arange(len(probabilities)) * window_step + window_size/2
        
        plt.plot(window_times, probabilities, 'r-', linewidth=2, label='Seizure Probability')
        plt.axhline(y=0.5, color='k', linestyle='--', label='Threshold')
        plt.axhline(y=0.85, color='orange', linestyle='--', label='High Confidence')
        plt.title('CNN-GRU Seizure Probability')
        plt.xlabel('Time (s)')
        plt.ylabel('Probability')
        plt.ylim([-0.05, 1.05])
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=100)
        plt.close()
        print(f"Visualization saved to: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_seizure.py <edf_file_path> [output_directory]")
        sys.exit(1)
    
    edf_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "EEG_Data_new"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    predictor = SeizurePredictor(output_dir=output_dir)
    result = predictor.predict(edf_file)
    
    print(f"Prediction for {edf_file}:")
    print(f"Result: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Visualization saved to: {result['visualization']}")

if __name__ == "__main__":
    main()