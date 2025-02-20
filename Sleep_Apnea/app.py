import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from modules.signal_processor import SignalProcessor
from modules.apnea_detector import ApneaDetector
from modules.snore_analyzer import SnoreAnalyzer
from modules.utils import get_recommendations

# Page configuration
st.set_page_config(
    page_title="Modeling Sleep Apnea",
    page_icon="🌙",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = None
if 'lstm_model_trained' not in st.session_state:
    st.session_state['lstm_model_trained'] = False
if 'detector' not in st.session_state:
    st.session_state['detector'] = ApneaDetector()

def show_respiratory_analysis():
    """Display and handle respiratory signal analysis section."""
    st.header("Respiratory Signal Analysis")

    uploaded_file = st.file_uploader("Upload respiratory signal data (CSV)",
                                   type=['csv'],
                                   key="respiratory_upload")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data

            st.success("Data uploaded successfully!")
            st.subheader("Data Preview")
            st.write(data.head())

            processor = SignalProcessor()
            if len(data.columns) > 1:
                signal_column = st.selectbox("Select respiratory signal column",
                                          data.columns)
                processed_signal = processor.apply_bandpass_filter(data[signal_column])
            else:
                processed_signal = processor.apply_bandpass_filter(data.iloc[:, 0])

            st.session_state['processed_data'] = processed_signal

            col1, col2 = st.columns(2)
            with col1:
                apnea_threshold = st.slider("Apnea Detection Threshold", 0.5, 0.9, 0.8, 0.05,
                                  help="Percent reduction in airflow for apnea detection")
                hypopnea_threshold = st.slider("Hypopnea Detection Threshold", 0.3, 0.7, 0.5, 0.05,
                                    help="Percent reduction in airflow for hypopnea detection")

            with col2:
                sampling_rate = st.number_input("Sampling Rate (Hz)", 10, 1000, 100)
                min_duration = st.number_input("Minimum Event Duration (seconds)",
                                            5, 20, 10)

            if st.button("Analyze Respiratory Signal"):
                with st.spinner("Analyzing..."):
                    detector = st.session_state['detector']
                    detector.sampling_rate = sampling_rate
                    detector.min_event_duration = min_duration

                    apnea_events, hypopnea_events, cleaned_signal = detector.detect_events(
                        processed_signal,
                        apnea_threshold=apnea_threshold,
                        hypopnea_threshold=hypopnea_threshold
                    )

                    # Save events for ML training
                    st.session_state['apnea_events'] = apnea_events
                    st.session_state['hypopnea_events'] = hypopnea_events
                    st.session_state['cleaned_signal'] = cleaned_signal

                    # Create visualization
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=cleaned_signal,
                        name="Respiratory Signal",
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        y=apnea_events * np.max(np.abs(cleaned_signal)),
                        name="Apnea Events",
                        line=dict(color='red')
                    ))
                    fig.add_trace(go.Scatter(
                        y=hypopnea_events * np.max(np.abs(cleaned_signal)) * 0.8,
                        name="Hypopnea Events",
                        line=dict(color='orange')
                    ))

                    fig.update_layout(
                        title="Respiratory Signal with Detected Apnea and Hypopnea Events",
                        xaxis_title="Time (samples)",
                        yaxis_title="Amplitude"
                    )
                    st.plotly_chart(fig)

                    # Calculate statistics
                    summary = detector.generate_summary(
                        apnea_events,
                        hypopnea_events,
                        len(cleaned_signal) / (sampling_rate * 3600)
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("AHI (events/hour)", f"{summary['ahi']:.1f}")
                    with col2:
                        st.metric("Apnea Events", summary['total_apnea'])
                    with col3:
                        st.metric("Hypopnea Events", summary['total_hypopnea'])
                    with col4:
                        st.metric("Severity", summary['severity'])

                    # Additional statistics
                    st.subheader("Detailed Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Apnea Event Statistics")
                        if summary['total_apnea'] > 0:
                            st.write(f"- Average duration: {summary['mean_apnea_duration']:.1f} seconds")
                            st.write(f"- Maximum duration: {summary['max_apnea_duration']:.1f} seconds")
                        else:
                            st.write("No apnea events detected")

                    with col2:
                        st.write("Hypopnea Event Statistics")
                        if summary['total_hypopnea'] > 0:
                            st.write(f"- Average duration: {summary['mean_hypopnea_duration']:.1f} seconds")
                            st.write(f"- Maximum duration: {summary['max_hypopnea_duration']:.1f} seconds")
                        else:
                            st.write("No hypopnea events detected")

        except Exception as e:
            st.error(f"Error processing respiratory data: {str(e)}")

def show_lstm_training():
    """Display and handle LSTM model training section."""
    st.header("LSTM Model Training")

    if 'cleaned_signal' not in st.session_state or 'apnea_events' not in st.session_state:
        st.warning("Please complete respiratory analysis first to generate data for model training")
        return

    col1, col2 = st.columns(2)
    with col1:
        sequence_length = st.slider("Sequence Length", 50, 300, 100, 10,
                               help="Number of timepoints in each training sequence")
        epochs = st.slider("Training Epochs", 5, 50, 10, 5)

    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        use_combined_events = st.checkbox("Use both apnea and hypopnea events for training", value=True)

    if st.button("Train LSTM Model"):
        with st.spinner("Training LSTM model... This may take a few minutes"):
            detector = st.session_state['detector']
            cleaned_signal = st.session_state['cleaned_signal']

            if use_combined_events and 'hypopnea_events' in st.session_state:
                events = np.logical_or(
                    st.session_state['apnea_events'],
                    st.session_state['hypopnea_events']
                ).astype(int)
            else:
                events = st.session_state['apnea_events']

            # Normalize signal for better training
            scaler = MinMaxScaler()
            normalized_signal = scaler.fit_transform(np.array(cleaned_signal).reshape(-1, 1)).flatten()

            # Train model
            history = detector.train_model(
                normalized_signal,
                events,
                sequence_length=sequence_length,
                epochs=epochs,
                batch_size=batch_size
            )

            st.session_state['lstm_model_trained'] = True

            # Plot training history
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history.history['accuracy'],
                name="Training Accuracy",
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                y=history.history['val_accuracy'],
                name="Validation Accuracy",
                line=dict(color='red')
            ))
            fig.update_layout(
                title="LSTM Training History",
                xaxis_title="Epoch",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig)

            # Make predictions
            predictions = detector.predict_with_lstm(normalized_signal, sequence_length)

            if predictions is not None:
                # Plot predictions vs actual
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=normalized_signal,
                    name="Respiratory Signal",
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=events * np.max(normalized_signal),
                    name="Actual Events",
                    line=dict(color='red')
                ))
                fig.add_trace(go.Scatter(
                    y=predictions * np.max(normalized_signal) * 0.8,
                    name="LSTM Predictions",
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title="LSTM Predictions vs Actual Events",
                    xaxis_title="Time (samples)",
                    yaxis_title="Normalized Amplitude"
                )
                st.plotly_chart(fig)

                # Calculate metrics
                acc = accuracy_score(events, predictions)
                prec = precision_score(events, predictions, zero_division=0)
                rec = recall_score(events, predictions, zero_division=0)
                f1 = f1_score(events, predictions, zero_division=0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{acc:.3f}")
                with col2:
                    st.metric("Precision", f"{prec:.3f}")
                with col3:
                    st.metric("Recall", f"{rec:.3f}")
                with col4:
                    st.metric("F1 Score", f"{f1:.3f}")

                st.success("LSTM model trained successfully! The model can now be used to predict apnea events on new data.")
            else:
                st.error("Failed to make predictions with the trained model.")
                
def show_snoring_analysis():
    """Display and handle snoring audio analysis section."""
    st.header("Snoring Audio Analysis")

    uploaded_file = st.file_uploader("Upload audio file",
                                   type=['wav', 'mp3', 'ogg', 'unknown'],
                                   key="audio_upload")

    if uploaded_file:
        with st.spinner("Processing audio..."):
            analyzer = SnoreAnalyzer()
            if analyzer.process_audio(uploaded_file):
                results = analyzer.analyze_snoring()

                if results:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Snores", results['total_snores'])
                    with col2:
                        st.metric("Average Interval (s)",
                                f"{results['mean_interval']:.2f}")
                    with col3:
                        st.metric("Snoring Rate (per minute)",
                                f"{results['snoring_rate']:.1f}")
                    with col4:
                        st.metric("Severity", results['severity'])

                    # Visualize snoring pattern
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=results['smoothed'],
                        name="Audio Envelope",
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=results['peaks'],
                        y=results['smoothed'][results['peaks']],
                        mode='markers',
                        name="Detected Snores",
                        marker=dict(color='red', size=10)
                    ))
                    fig.update_layout(title="Snoring Pattern Analysis")
                    st.plotly_chart(fig)

                    # Show interval distribution
                    if len(results['intervals']) > 0:
                        fig = px.histogram(
                            results['intervals'],
                            title="Snoring Interval Distribution",
                            labels={'value': 'Interval (seconds)', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig)

                    return results
            else:
                st.error("Failed to process audio file. Please check the format and try again.")
                
def show_recommendations(apnea_summary=None, snoring_results=None):
    """Display overall assessment and recommendations."""
    st.header("Analysis Summary and Recommendations")

    if apnea_summary or snoring_results:
        apnea_severity = apnea_summary['severity'] if apnea_summary else "Normal"
        snoring_severity = snoring_results['severity'] if snoring_results else "Low"

        st.subheader("Overall Assessment")

        col1, col2 = st.columns(2)
        with col1:
            st.write("Apnea Assessment:")
            st.write(f"- Severity: {apnea_severity}")
            if apnea_summary:
                st.write(f"- AHI: {apnea_summary['ahi']:.1f} events/hour")
                st.write(f"- Total events: {apnea_summary['total_events']}")

        with col2:
            st.write("Snoring Assessment:")
            st.write(f"- Severity: {snoring_severity}")
            if snoring_results:
                st.write(f"- Rate: {snoring_results['snoring_rate']:.1f} per minute")
                st.write(f"- Total snores: {snoring_results['total_snores']}")

        st.subheader("Recommendations")
        recommendations = get_recommendations(apnea_severity, snoring_severity)
        st.markdown(recommendations)
    else:
        st.info("Complete both respiratory and snoring analysis to get personalized recommendations")

def main():
    """Main application entry point."""
    st.title("Modeling Sleep Apnea: A Comprehensive Detection and Analysis System 🌙")
    st.write("Analyze respiratory signals and audio for sleep apnea and snoring detection")

    tabs = st.tabs(["Respiratory Analysis", "ML Analysis", "Snoring Analysis", "Summary & Recommendations"])

    with tabs[0]:
        show_respiratory_analysis()

    with tabs[1]:
        show_lstm_training()

    with tabs[2]:
        snoring_results = show_snoring_analysis()
        if snoring_results:
            st.session_state['snoring_results'] = snoring_results

    with tabs[3]:
        apnea_summary = None
        snoring_results = None

        if 'processed_data' in st.session_state and st.session_state['processed_data'] is not None:
            detector = st.session_state['detector']
            apnea_events, hypopnea_events, _ = detector.detect_events(st.session_state['processed_data'])
            apnea_summary = detector.generate_summary(
                apnea_events,
                hypopnea_events,
                len(st.session_state['processed_data']) / (100 * 3600)  # Assuming 100Hz sampling rate
            )

        if 'snoring_results' in st.session_state:
            snoring_results = st.session_state['snoring_results']

        show_recommendations(apnea_summary, snoring_results)

if __name__ == "__main__":
    main()