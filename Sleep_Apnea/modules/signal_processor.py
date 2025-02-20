import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import tsfel

class SignalProcessor:
    """
    Provides signal processing functionality for respiratory signals.
    
    This class handles filtering, preprocessing, and feature extraction from 
    respiratory signals for sleep apnea detection.
    """
    
    def __init__(self, sampling_rate=100):
        """
        Initialize the signal processor.
        
        Args:
            sampling_rate (int): The sampling rate of the signal in Hz. Default is 100 Hz.
        """
        self.sampling_rate = sampling_rate

    def butter_bandpass(self, lowcut=0.1, highcut=0.5, order=2):
        """
        Design a Butterworth bandpass filter.
        
        Args:
            lowcut (float): Lower cutoff frequency in Hz. Default is 0.1 Hz.
            highcut (float): Upper cutoff frequency in Hz. Default is 0.5 Hz.
            order (int): Filter order. Default is 2.
            
        Returns:
            tuple: Filter coefficients (b, a)
        """
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_bandpass_filter(self, data):
        """
        Apply bandpass filter to respiratory signal data.
        
        Args:
            data (array-like): The respiratory signal data.
            
        Returns:
            ndarray: Filtered signal
        """
        b, a = self.butter_bandpass()
        return filtfilt(b, a, data)

    def remove_baseline_wander(self, data, window_size=100):
        """
        Remove baseline wander from the signal using rolling mean.
        
        Args:
            data (array-like): The respiratory signal data.
            window_size (int): Size of the rolling window in samples. Default is 100.
            
        Returns:
            ndarray: Signal with baseline wander removed
        """
        baseline = pd.Series(data).rolling(window=window_size, center=True).mean()
        return data - baseline.fillna(method='bfill').fillna(method='ffill')

    def extract_respiratory_features(self, data):
        """
        Extract time series features from respiratory signal.
        
        Args:
            data (array-like): The respiratory signal data.
            
        Returns:
            DataFrame: Extracted features
        """
        cfg = tsfel.get_features_by_domain()
        features = tsfel.time_series_features_extractor(cfg, data.reshape(-1, 1))
        return features

    def prepare_sequences(self, data, sequence_length=100, step=20):
        """
        Prepare data for sequence modeling (LSTM).
        
        Args:
            data (array-like): The respiratory signal data.
            sequence_length (int): Length of each sequence. Default is 100.
            step (int): Step size between sequences. Default is 20.
            
        Returns:
            ndarray: Array of sequences
        """
        sequences = []
        for i in range(0, len(data) - sequence_length, step):
            sequences.append(data[i:i+sequence_length])
        return np.array(sequences)