# Sleep Apnea Detection and Analysis System

A comprehensive tool for detecting and analyzing sleep apnea through respiratory signals and snoring audio analysis.

## Overview

This application provides a research-grade solution for sleep apnea detection following American Academy of Sleep Medicine (AASM) guidelines. It combines traditional signal processing techniques with deep learning to offer an accessible alternative to polysomnography.

![image](https://github.com/user-attachments/assets/ad0dc389-e1c1-46b5-830c-ab80286629cc)


## Features

- **Respiratory Signal Analysis**
  - Implements AASM thresholds (≥80% reduction for apnea, ≥30% reduction for hypopnea)
  - Automated event detection with configurable parameters
  - Comprehensive clinical metrics (AHI, event duration, severity classification)

- **Machine Learning Analysis**
  - LSTM-based sequence modeling for improved detection
  - Class-balanced training to handle imbalanced sleep data
  - Performance evaluation with healthcare-specific metrics

- **Snoring Analysis**
  - Multi-format audio support
  - Adaptive threshold-based event detection
  - Statistical analysis of snoring patterns

- **Comprehensive Reporting**
  - Severity classification following clinical guidelines
  - Combined assessment of respiratory and audio analyses
  - Personalized recommendations

## Screenshots

### Respiratory Analysis
![image](https://github.com/user-attachments/assets/5dc5b657-12cf-4991-ae43-f97c20138e47)
![image](https://github.com/user-attachments/assets/fe2c4ccc-2ce9-4b07-bcc3-d6030ad75f14)
![image](https://github.com/user-attachments/assets/55965dd9-d3fd-443b-88e5-e10ab052b7a8)

*(Image placeholder - will be updated)*

### LSTM Model Training
![image](https://github.com/user-attachments/assets/3a75899b-25ce-4202-af1c-83a4fb42e12c)

### Snoring Analysis
![image](https://github.com/user-attachments/assets/d30e52b3-a55e-4a74-ae7b-b9cfc82c4fa0)
![image](https://github.com/user-attachments/assets/e90a46fb-9a27-4f20-9961-ede16cfe8a1b)


### Recommendations
![image](https://github.com/user-attachments/assets/cbd7314d-c28d-4607-99f7-2bc480df451f)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sleep-apnea-detection.git
cd sleep-apnea-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
streamlit run app.py
```


## Technical Implementation

### Signal Processing Pipeline
- Butterworth bandpass filter (0.1-0.5 Hz)
- Adaptive baseline wander removal
- Feature extraction using TSFEL

### LSTM Network Architecture
```
Model: Sequential
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_layer (InputLayer)    [(None, 100, 1)]         0         
                                                                 
 lstm_1 (LSTM)               (None, 100, 64)          16896     
                                                                 
 dropout_1 (Dropout)         (None, 100, 64)          0         
                                                                 
 lstm_2 (LSTM)               (None, 32)               12416     
                                                                 
 dropout_2 (Dropout)         (None, 32)               0         
                                                                 
 dense_1 (Dense)             (None, 16)               528       
                                                                 
 dense_2 (Dense)             (None, 1)                17        
=================================================================
Total params: 29,857
Trainable params: 29,857
Non-trainable params: 0
```


## Future Directions

- Clinical validation against polysomnography
- Real-time monitoring capabilities
- Edge device optimization
- Integration with wearable sensors
- Longitudinal analysis functionality

## Contributors

- Gandham Mani Saketh
