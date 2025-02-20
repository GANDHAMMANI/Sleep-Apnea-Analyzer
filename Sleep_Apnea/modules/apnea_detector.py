import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from modules.signal_processor import SignalProcessor

class ApneaDetector:
    """
    Detects sleep apnea and hypopnea events in respiratory signals.
    
    Implements both rule-based detection following AASM guidelines and
    machine learning-based detection using LSTM neural networks.
    """
    
    def __init__(self, sampling_rate=100):
        """
        Initialize the apnea detector.
        
        Args:
            sampling_rate (int): The sampling rate of the signal in Hz. Default is 100 Hz.
        """
        self.sampling_rate = sampling_rate
        self.min_event_duration = 10  # minimum duration in seconds
        self.processor = SignalProcessor(sampling_rate)
        self.model = None

    def detect_events(self, respiratory_signal, apnea_threshold=0.3, hypopnea_threshold=0.15):
        """
        Detect apnea and hypopnea events in the respiratory signal.
        
        Args:
            respiratory_signal (array-like): The respiratory signal data.
            apnea_threshold (float): Threshold for apnea detection (>80% reduction).
                Default is 0.3 (normalized).
            hypopnea_threshold (float): Threshold for hypopnea detection (>30% reduction).
                Default is 0.15 (normalized).
                
        Returns:
            tuple: (apnea_events, hypopnea_events, cleaned_signal)
        """
        filtered_signal = self.processor.apply_bandpass_filter(respiratory_signal)
        cleaned_signal = self.processor.remove_baseline_wander(filtered_signal)

        baseline = np.mean(np.abs(cleaned_signal))
        reduction = 1 - (np.abs(cleaned_signal) / baseline)

        # Detect apnea events (>80% reduction)
        apnea_events = reduction > apnea_threshold

        # Detect hypopnea events (30-50% reduction)
        hypopnea_events = (reduction > hypopnea_threshold) & (reduction <= apnea_threshold)

        min_samples = self.min_event_duration * self.sampling_rate

        # Process apnea events
        labeled_apnea = self.label_continuous_events(apnea_events)
        valid_apnea = np.zeros_like(apnea_events)
        for event_id in range(1, labeled_apnea.max() + 1) if labeled_apnea.max() > 0 else []:
            event_mask = labeled_apnea == event_id
            if np.sum(event_mask) >= min_samples:
                valid_apnea[event_mask] = 1

        # Process hypopnea events
        labeled_hypopnea = self.label_continuous_events(hypopnea_events)
        valid_hypopnea = np.zeros_like(hypopnea_events)
        for event_id in range(1, labeled_hypopnea.max() + 1) if labeled_hypopnea.max() > 0 else []:
            event_mask = labeled_hypopnea == event_id
            if np.sum(event_mask) >= min_samples:
                valid_hypopnea[event_mask] = 1

        return valid_apnea, valid_hypopnea, cleaned_signal

    def label_continuous_events(self, binary_events):
        """
        Label continuous segments of events with unique IDs.
        
        Args:
            binary_events (array-like): Binary array indicating event/non-event.
            
        Returns:
            ndarray: Array with continuous events labeled with unique IDs
        """
        labeled_events = np.zeros_like(binary_events, dtype=int)
        current_label = 0
        in_event = False

        for i in range(len(binary_events)):
            if binary_events[i] and not in_event:
                current_label += 1
                in_event = True
            elif not binary_events[i]:
                in_event = False

            if in_event:
                labeled_events[i] = current_label

        return labeled_events

    def calculate_ahi(self, apnea_events, hypopnea_events, recording_duration_hours):
        """
        Calculate Apnea-Hypopnea Index (AHI).
        
        Args:
            apnea_events (array-like): Binary array indicating apnea events.
            hypopnea_events (array-like): Binary array indicating hypopnea events.
            recording_duration_hours (float): Duration of recording in hours.
            
        Returns:
            tuple: (ahi, total_apnea_events, total_hypopnea_events)
        """
        total_apnea_events = np.sum(np.diff(apnea_events) > 0)
        total_hypopnea_events = np.sum(np.diff(hypopnea_events) > 0)
        total_events = total_apnea_events + total_hypopnea_events
        ahi = total_events / recording_duration_hours
        return ahi, total_apnea_events, total_hypopnea_events

    def analyze_severity(self, ahi):
        """
        Determine severity based on AHI.
        
        Args:
            ahi (float): Apnea-Hypopnea Index value.
            
        Returns:
            str: Severity classification ("Normal", "Mild", "Moderate", or "Severe")
        """
        if ahi < 5:
            return "Normal"
        elif ahi < 15:
            return "Mild"
        elif ahi < 30:
            return "Moderate"
        else:
            return "Severe"

    def get_event_durations(self, events):
        """
        Calculate durations of detected events.
        
        Args:
            events (array-like): Binary array indicating events.
            
        Returns:
            list: List of event durations in seconds
        """
        durations = []
        current_duration = 0

        for i in range(len(events)):
            if events[i]:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration / self.sampling_rate)
                current_duration = 0

        if current_duration > 0:
            durations.append(current_duration / self.sampling_rate)

        return durations

    def generate_summary(self, apnea_events, hypopnea_events, recording_duration_hours):
        """
        Generate summary statistics for the detected events.
        
        Args:
            apnea_events (array-like): Binary array indicating apnea events.
            hypopnea_events (array-like): Binary array indicating hypopnea events.
            recording_duration_hours (float): Duration of recording in hours.
            
        Returns:
            dict: Dictionary containing summary statistics
        """
        ahi, total_apnea, total_hypopnea = self.calculate_ahi(apnea_events, hypopnea_events, recording_duration_hours)
        severity = self.analyze_severity(ahi)

        apnea_durations = self.get_event_durations(apnea_events)
        hypopnea_durations = self.get_event_durations(hypopnea_events)

        return {
            'ahi': ahi,
            'severity': severity,
            'total_apnea': total_apnea,
            'total_hypopnea': total_hypopnea,
            'total_events': total_apnea + total_hypopnea,
            'mean_apnea_duration': np.mean(apnea_durations) if apnea_durations else 0,
            'mean_hypopnea_duration': np.mean(hypopnea_durations) if hypopnea_durations else 0,
            'max_apnea_duration': np.max(apnea_durations) if apnea_durations else 0,
            'max_hypopnea_duration': np.max(hypopnea_durations) if hypopnea_durations else 0,
            'recording_hours': recording_duration_hours
        }

    def build_lstm_model(self, sequence_length, pos_weight=10):
        """
        Build LSTM model with weighted loss to address class imbalance.
        
        Args:
            sequence_length (int): Length of input sequences.
            pos_weight (float): Weight for positive class in loss function. Default is 10.
            
        Returns:
            Model: Compiled Keras model
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=(sequence_length, 1)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        # Use weighted binary crossentropy to give more importance to positive class
        def weighted_binary_crossentropy(y_true, y_pred):
            weights = (y_true * (pos_weight - 1)) + 1
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            weighted_bce = weights * bce
            return tf.reduce_mean(weighted_bce)

        model.compile(
            optimizer='adam',
            loss=weighted_binary_crossentropy,
            metrics=['accuracy',
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   tf.keras.metrics.AUC()]
        )

        return model

    def prepare_balanced_training_data(self, signal, events, sequence_length=100, step=20):
        """
        Prepare balanced training data with appropriate positive and negative examples.
        
        Args:
            signal (array-like): The respiratory signal data.
            events (array-like): Binary array indicating events.
            sequence_length (int): Length of each sequence. Default is 100.
            step (int): Step size between sequences. Default is 20.
            
        Returns:
            tuple: (X, y) balanced training data and labels
        """
        X_pos = []
        y_pos = []
        X_neg = []
        y_neg = []

        for i in range(0, len(signal) - sequence_length, step):
            sequence = signal[i:i+sequence_length]
            # Label the sequence based on if any points are events (more sensitive)
            event_ratio = np.mean(events[i:i+sequence_length])

            # Consider sequence positive if it contains significant event ratio
            if event_ratio > 0.2:  # More sensitive threshold
                X_pos.append(sequence.reshape(-1, 1))
                y_pos.append(1)
            else:
                X_neg.append(sequence.reshape(-1, 1))
                y_neg.append(0)

        print(f"Original distribution - Positive: {len(X_pos)}, Negative: {len(X_neg)}")

        # Handle case with no positive samples
        if len(X_pos) == 0:
            print("Warning: No positive samples detected. Using original imbalanced data.")
            return np.array(X_neg), np.array(y_neg)

        # Balance dataset by undersampling majority class
        n_samples = min(len(X_pos) * 3, len(X_neg))  # Keep ratio 1:3 positive:negative

        # Randomly sample from negative class
        neg_indices = np.random.choice(len(X_neg), n_samples, replace=False)
        X_neg_sampled = [X_neg[i] for i in neg_indices]
        y_neg_sampled = [y_neg[i] for i in neg_indices]

        # Combine positive and negative examples
        X = X_pos + X_neg_sampled
        y = y_pos + y_neg_sampled

        # Shuffle
        combined = list(zip(X, y))
        np.random.shuffle(combined)
        X, y = zip(*combined)

        print(f"Balanced distribution - Positive: {len(X_pos)}, Negative: {len(X_neg_sampled)}")
        return np.array(X), np.array(y)

    def train_model(self, signal, events, sequence_length=100, epochs=10, batch_size=32):
        """
        Train LSTM model on signal data and events.
        
        Args:
            signal (array-like): The respiratory signal data.
            events (array-like): Binary array indicating events.
            sequence_length (int): Length of each sequence. Default is 100.
            epochs (int): Number of training epochs. Default is 10.
            batch_size (int): Batch size for training. Default is 32.
            
        Returns:
            History: Training history object
        """
        # Prepare balanced data
        X, y = self.prepare_balanced_training_data(signal, events, sequence_length)

        # Check if we have any data
        if len(X) == 0:
            print("No training data available.")
            return None

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build model with higher class weight
        self.model = self.build_lstm_model(sequence_length, pos_weight=20)

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        return history

    def predict_with_lstm(self, signal, sequence_length=100, step=50, threshold=0.3):
        """
        Make predictions using trained LSTM model.
        
        Args:
            signal (array-like): The respiratory signal data.
            sequence_length (int): Length of each sequence. Default is 100.
            step (int): Step size between sequences. Default is 50.
            threshold (float): Prediction threshold. Default is 0.3.
            
        Returns:
            ndarray: Binary predictions array
        """
        if self.model is None:
            return None

        # Prepare sequences
        sequences = []
        for i in range(0, len(signal) - sequence_length, step):
            sequences.append(signal[i:i+sequence_length].reshape(-1, 1))

        if not sequences:
            return None

        # Make predictions
        pred_sequences = np.array(sequences)
        predictions = self.model.predict(pred_sequences)

        # Map predictions back to original signal length
        full_predictions = np.zeros(len(signal))
        count = np.zeros(len(signal))

        for i, pred in enumerate(predictions):
            idx = i * step
            full_predictions[idx:idx+sequence_length] += pred[0]
            count[idx:idx+sequence_length] += 1

        # Average overlapping predictions
        mask = count > 0
        full_predictions[mask] /= count[mask]

        # Use a lower threshold (0.3 instead of 0.5) to improve recall
        thresholded = (full_predictions > threshold).astype(int)

        return thresholded