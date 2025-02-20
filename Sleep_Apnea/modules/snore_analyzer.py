import os
import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment
from scipy.signal import find_peaks

class SnoreAnalyzer:
    """
    Analyzes audio recordings for snoring detection and severity assessment.
    
    Provides methods for processing audio files and analyzing snoring patterns.
    """
    
    def __init__(self):
        """
        Initialize the snore analyzer.
        """
        self.audio = None
        self.sr = None

    def process_audio(self, audio_file):
        """
        Process audio file for analysis.
        
        Args:
            audio_file: Uploaded audio file (supports various formats).
            
        Returns:
            bool: True if processing successful, False otherwise.
        """
        try:
            audio = AudioSegment.from_file(audio_file)
            temp_wav = "temp_audio.wav"
            audio.export(temp_wav, format="wav")
            self.audio, self.sr = librosa.load(temp_wav)
            os.remove(temp_wav)
            return True
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return False

    def analyze_snoring(self):
        """
        Analyze audio for snoring patterns.
        
        This method applies signal processing to identify snoring events
        and calculate statistics about their frequency and patterns.
        
        Returns:
            dict: Dictionary containing analysis results or None if no audio loaded.
        """
        if self.audio is None:
            return None

        # Calculate amplitude envelope
        envelope = np.abs(self.audio)
        
        # Smooth the envelope
        window_size = int(self.sr * 0.1)
        smoothed = pd.Series(envelope).rolling(window_size, center=True).mean()
        smoothed = smoothed.fillna(method='bfill').fillna(method='ffill')

        # Detect peaks (potential snores)
        peaks, _ = find_peaks(smoothed,
                             height=np.mean(smoothed) + np.std(smoothed),
                             distance=int(self.sr * 0.5))

        # Calculate intervals between detected snores
        intervals = np.diff(peaks) / self.sr

        # Determine severity based on snoring rate
        severity = "Low"
        if len(peaks) > 0:
            snoring_rate = 60 / np.mean(intervals) if len(intervals) > 0 else 0
            if snoring_rate > 20:
                severity = "High"
            elif snoring_rate > 10:
                severity = "Moderate"

        # Return analysis results
        return {
            'smoothed': smoothed,
            'peaks': peaks,
            'intervals': intervals,
            'total_snores': len(peaks),
            'mean_interval': np.mean(intervals) if len(intervals) > 0 else 0,
            'snoring_rate': 60 / np.mean(intervals) if len(intervals) > 0 else 0,
            'duration': len(self.audio) / self.sr,
            'severity': severity
        }