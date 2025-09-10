import streamlit as st
import torch
import numpy as np
from transformers import AutoModel, AutoProcessor
import librosa
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import io
import matplotlib.pyplot as plt
import seaborn as sns

class MERTAudioAnalyzer:
    def __init__(self):
        self.model_id = "m-a-p/MERT-v1-330M"
        self.max_duration = 30.0
        self.genre_classifier = None
        self.setup_model()
        
        # Define genre labels
        self.genres = ['rock', 'pop', 'hip hop', 'classical', 'jazz', 
                      'electronic', 'country', 'metal', 'blues', 'reggae']
    
    @st.cache_resource
    def setup_model(_self):
        """Initialize the MERT model and processor"""
        _self.processor = AutoProcessor.from_pretrained(_self.model_id, trust_remote_code=True)
        _self.model = AutoModel.from_pretrained(_self.model_id, trust_remote_code=True)
        _self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _self.model = _self.model.to(_self.device)
        _self.model.eval()
    
    def train_genre_classifier(self, features, labels):
        """Train a simple genre classifier"""
        self.genre_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.genre_classifier.fit(features_scaled, labels)
    
    def predict_genre(self, features):
        """Predict genre for given features"""
        if self.genre_classifier is None:
            return None
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.reshape(1, -1))
        prediction = self.genre_classifier.predict(features_scaled)
        probabilities = self.genre_classifier.predict_proba(features_scaled)
        
        return prediction[0], probabilities[0]

    def load_audio(self, audio_file, max_duration=None):
        """Load and preprocess audio file"""
        if max_duration is None:
            max_duration = self.max_duration
        
        y, sr = librosa.load(audio_file, sr=24000)
        
        if len(y) > int(sr * max_duration):
            y = y[:int(sr * max_duration)]
        
        return y, sr
    
    def extract_features(self, audio_file):
        """Extract features using MERT model"""
        y, sr = self.load_audio(audio_file)
        
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features.cpu().numpy(), y, sr
    
    def analyze_audio_properties(self, y, sr):
        """Extract various audio properties"""
        properties = {}
        
        properties['duration'] = len(y) / sr
        properties['sample_rate'] = sr
        properties['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        properties['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        properties['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        properties['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        properties['tempo'] = tempo
        properties['rms_energy'] = np.mean(librosa.feature.rms(y=y))
        
        # Chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        properties['chroma_mean'] = np.mean(chroma)
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(5):
            properties[f'mfcc_{i+1}'] = np.mean(mfcc[i])
        
        return properties
    
    def compare_audio_files(self, features_list):
        """Compare multiple audio files using cosine similarity"""
        if len(features_list) < 2:
            return None
        
        similarities = []
        for i in range(len(features_list)):
            row_similarities = []
            for j in range(len(features_list)):
                if i == j:
                    similarity = 1.0
                else:
                    similarity = cosine_similarity(
                        features_list[i].reshape(1, -1),
                        features_list[j].reshape(1, -1)
                    )[0][0]
                row_similarities.append(similarity)
            similarities.append(row_similarities)
        
        return np.array(similarities)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="MERT Audio Analyzer",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 MERT Audio Analysis Suite")
    st.markdown("Advanced audio analysis using the MERT-v1-330M model")

    @st.cache_resource
    def load_analyzer():
        return MERTAudioAnalyzer()

    analyzer = load_analyzer()

    # Task selection
    task = st.sidebar.selectbox(
        "Choose Task:",
        ["Single Audio Analysis", "Genre Classification", "Audio Comparison", "Batch Processing"]
    )

    if task == "Single Audio Analysis":
        single_audio_analysis(analyzer)
    elif task == "Genre Classification":
        genre_classification(analyzer)
    elif task == "Audio Comparison":
        audio_comparison(analyzer)
    elif task == "Batch Processing":
        batch_processing(analyzer)

def single_audio_analysis(analyzer):
    st.header("Single Audio Analysis")
    
    audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'ogg'])
    
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            temp_path = tmp_file.name

        try:
            st.audio(audio_file)
            
            features, y, sr = analyzer.extract_features(temp_path)
            properties = analyzer.analyze_audio_properties(y, sr)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Audio Properties")
                props_df = pd.DataFrame(properties.items(), columns=['Property', 'Value'])
                st.dataframe(props_df)
                
                # Waveform
                fig_wave = px.line(y=y[:10000], title="Waveform")
                st.plotly_chart(fig_wave)

            with col2:
                st.subheader("MERT Features")
                fig_features = px.line(features[0], title="Feature Vector")
                st.plotly_chart(fig_features)
                
                # Mel Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                fig_mel = px.imshow(librosa.power_to_db(mel_spec), 
                                  aspect='auto', 
                                  title="Mel Spectrogram")
                st.plotly_chart(fig_mel)

        finally:
            os.remove(temp_path)

def genre_classification(analyzer):
    st.header("Genre Classification")
    
    # Upload training data if classifier not trained
    if analyzer.genre_classifier is None:
        st.info("Upload training examples for each genre to train the classifier")
        
        training_files = st.file_uploader("Upload training audio files", 
                                        type=['wav', 'mp3', 'ogg'],
                                        accept_multiple_files=True)
        
        genre_labels = st.multiselect("Select genres for uploaded files", 
                                    analyzer.genres,
                                    default=analyzer.genres[:len(training_files)])
        
        if training_files and len(genre_labels) == len(training_files):
            features_list = []
            
            for audio_file in training_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    features, _, _ = analyzer.extract_features(tmp_file.name)
                    features_list.append(features[0])
                    os.remove(tmp_file.name)
            
            analyzer.train_genre_classifier(np.array(features_list), genre_labels)
            st.success("Genre classifier trained successfully!")
    
    # Genre prediction
    audio_file = st.file_uploader("Upload audio for genre classification", 
                                 type=['wav', 'mp3', 'ogg'])
    
    if audio_file and analyzer.genre_classifier:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            
            features, _, _ = analyzer.extract_features(tmp_file.name)
            predicted_genre, probabilities = analyzer.predict_genre(features[0])
            
            st.audio(audio_file)
            st.subheader(f"Predicted Genre: {predicted_genre}")
            
            # Genre probabilities
            fig = px.bar(x=analyzer.genres, y=probabilities,
                        title="Genre Probabilities",
                        labels={'x': 'Genre', 'y': 'Probability'})
            st.plotly_chart(fig)
            
            os.remove(tmp_file.name)

def audio_comparison(analyzer):
    st.header("Audio Comparison")
    
    files = st.file_uploader("Upload multiple audio files", 
                            type=['wav', 'mp3', 'ogg'],
                            accept_multiple_files=True)
    
    if len(files) >= 2:
        features_list = []
        file_names = []
        
        for audio_file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                features, _, _ = analyzer.extract_features(tmp_file.name)
                features_list.append(features[0])
                file_names.append(audio_file.name)
                os.remove(tmp_file.name)
        
        similarities = analyzer.compare_audio_files(features_list)
        
        fig = px.imshow(similarities,
                       x=file_names,
                       y=file_names,
                       title="Audio Similarity Matrix")
        st.plotly_chart(fig)

def batch_processing(analyzer):
    st.header("Batch Processing")
    
    files = st.file_uploader("Upload multiple audio files", 
                            type=['wav', 'mp3', 'ogg'],
                            accept_multiple_files=True)
    
    if files:
        results = []
        
        for audio_file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                
                features, y, sr = analyzer.extract_features(tmp_file.name)
                properties = analyzer.analyze_audio_properties(y, sr)
                
                result = {
                    'filename': audio_file.name,
                    **properties,
                    'feature_mean': np.mean(features),
                    'feature_std': np.std(features)
                }
                
                if analyzer.genre_classifier:
                    genre, _ = analyzer.predict_genre(features[0])
                    result['predicted_genre'] = genre
                
                results.append(result)
                os.remove(tmp_file.name)
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Results CSV",
            csv,
            "audio_analysis_results.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()