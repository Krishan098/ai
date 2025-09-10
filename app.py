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
import io
import matplotlib.pyplot as plt
import seaborn as sns

class MERTAudioAnalyzer:
    def __init__(self):
        self.model_id = "m-a-p/MERT-v1-95M"
        self.max_duration = 30.0
        self.setup_model()
    
    @st.cache_resource
    def setup_model(_self):
        """Initialize the MERT model and processor"""
        _self.processor = AutoProcessor.from_pretrained(_self.model_id, trust_remote_code=True)
        _self.model = AutoModel.from_pretrained(_self.model_id, trust_remote_code=True)
        _self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _self.model = _self.model.to(_self.device)
        _self.model.eval()
    
    def load_audio(self, audio_file, max_duration=None):
        """Load and preprocess audio file"""
        if max_duration is None:
            max_duration = self.max_duration
        
        # Load audio at 24kHz (MERT's expected sample rate)
        y, sr = librosa.load(audio_file, sr=24000)
        
        # Truncate if longer than max_duration
        if len(y) > int(sr * max_duration):
            y = y[:int(sr * max_duration)]
        
        return y, sr
    
    def extract_features(self, audio_file):
        """Extract features using MERT model"""
        y, sr = self.load_audio(audio_file)
        
        # Process audio through MERT
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get the last hidden state as features
            features = outputs.last_hidden_state.mean(dim=1)  # Average pooling
        
        return features.cpu().numpy(), y, sr
    
    def analyze_audio_properties(self, y, sr):
        """Extract various audio properties"""
        properties = {}
        
        # Basic properties
        properties['duration'] = len(y) / sr
        properties['sample_rate'] = sr
        
        # Spectral features
        properties['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        properties['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        properties['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        properties['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        properties['tempo'] = tempo
        
        # Energy features
        properties['rms_energy'] = np.mean(librosa.feature.rms(y=y))
        
        # MFCC features (first few coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(5):  # First 5 MFCC coefficients
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
st.set_page_config(
    page_title="MERT Audio Analyzer",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 MERT Audio Analysis Suite")
st.markdown("Advanced audio analysis using the MERT-v1-330M model for music understanding")

# Initialize analyzer
@st.cache_resource
def load_analyzer():
    try:
        return MERTAudioAnalyzer()
    except Exception as e:
        st.error("Failed to load MERT model. This might be due to missing dependencies.")
        st.code("""
# Install required dependencies:
pip install nnAudio==0.3.2
pip install torch torchaudio
pip install transformers>=4.21.0
""")
        st.stop()

# Show installation instructions
st.info("""
📋 **Required Dependencies for MERT:**
```bash
pip install nnAudio==0.3.2 torch torchaudio transformers>=4.21.0 librosa plotly pandas scikit-learn
```
""")

with st.spinner("Loading MERT model... (This may take a few minutes on first run)"):
    analyzer = load_analyzer()

st.success("✅ MERT model loaded successfully!")

# Sidebar for task selection
st.sidebar.title("Analysis Tasks")
task = st.sidebar.selectbox(
    "Choose an analysis task:",
    [
        "Single Audio Analysis",
        "Audio Feature Extraction",
        "Audio Comparison",
        "Batch Audio Processing"
    ]
)

if task == "Single Audio Analysis":
    st.header("📊 Single Audio File Analysis")
    st.write("Upload an audio file to get comprehensive analysis including features and properties.")
    
    audio_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        key="single_audio"
    )
    
    if audio_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Display audio player
            st.audio(audio_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔍 Analyze Audio", type="primary"):
                    with st.spinner("Extracting features and analyzing..."):
                        # Extract MERT features
                        features, y, sr = analyzer.extract_features(temp_path)
                        
                        # Analyze audio properties
                        properties = analyzer.analyze_audio_properties(y, sr)
                        
                        st.subheader("🎯 MERT Features")
                        st.write(f"Feature vector shape: {features.shape}")
                        st.write(f"Feature vector norm: {np.linalg.norm(features):.4f}")
                        
                        # Display feature visualization
                        fig = px.line(
                            y=features[0][:100],  # Show first 100 dimensions
                            title="MERT Feature Vector (First 100 dimensions)",
                            labels={'index': 'Feature Dimension', 'y': 'Feature Value'}
                        )
                        st.plotly_chart(fig)
            
            with col2:
                if 'properties' in locals():
                    st.subheader("🎼 Audio Properties")
                    
                    # Create a properties dataframe for better display
                    props_df = pd.DataFrame([
                        {"Property": "Duration", "Value": f"{properties['duration']:.2f} seconds"},
                        {"Property": "Sample Rate", "Value": f"{properties['sample_rate']} Hz"},
                        {"Property": "Tempo", "Value": f"{properties['tempo']:.1f} BPM"},
                        {"Property": "Spectral Centroid", "Value": f"{properties['spectral_centroid']:.1f} Hz"},
                        {"Property": "Spectral Bandwidth", "Value": f"{properties['spectral_bandwidth']:.1f} Hz"},
                        {"Property": "Zero Crossing Rate", "Value": f"{properties['zero_crossing_rate']:.4f}"},
                        {"Property": "RMS Energy", "Value": f"{properties['rms_energy']:.4f}"},
                    ])
                    st.dataframe(props_df, hide_index=True)
                    
                    # MFCC visualization
                    mfcc_values = [properties[f'mfcc_{i+1}'] for i in range(5)]
                    fig_mfcc = px.bar(
                        x=[f'MFCC {i+1}' for i in range(5)],
                        y=mfcc_values,
                        title="First 5 MFCC Coefficients"
                    )
                    st.plotly_chart(fig_mfcc)
        
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

elif task == "Audio Feature Extraction":
    st.header("🔢 Audio Feature Extraction")
    st.write("Extract and download MERT features from audio files.")
    
    audio_files = st.file_uploader(
        "Choose audio files",
        type=['wav', 'mp3', 'flac', 'm4a'],
        accept_multiple_files=True,
        key="feature_extraction"
    )
    
    if audio_files:
        st.write(f"Selected {len(audio_files)} files")
        
        if st.button("📥 Extract Features", type="primary"):
            features_data = {}
            progress_bar = st.progress(0)
            
            for i, audio_file in enumerate(audio_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    with st.spinner(f"Processing {audio_file.name}..."):
                        features, _, _ = analyzer.extract_features(temp_path)
                        features_data[audio_file.name] = features[0]
                    
                    progress_bar.progress((i + 1) / len(audio_files))
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            st.success("✅ Feature extraction completed!")
            
            # Create downloadable CSV
            features_df = pd.DataFrame(features_data).T  # Transpose so files are rows
            csv = features_df.to_csv()
            
            st.download_button(
                label="💾 Download Features as CSV",
                data=csv,
                file_name="mert_features.csv",
                mime="text/csv"
            )
            
            # Show summary statistics
            st.subheader("📈 Feature Summary")
            st.write(f"Number of files: {len(audio_files)}")
            st.write(f"Feature dimensions: {features_df.shape[1]}")
            st.dataframe(features_df.describe())

elif task == "Audio Comparison":
    st.header("🔄 Audio File Comparison")
    st.write("Compare multiple audio files using MERT features and cosine similarity.")
    
    audio_files = st.file_uploader(
        "Choose audio files to compare (2-10 files)",
        type=['wav', 'mp3', 'flac', 'm4a'],
        accept_multiple_files=True,
        key="audio_comparison"
    )
    
    if audio_files and len(audio_files) >= 2:
        st.write(f"Selected {len(audio_files)} files for comparison")
        
        if len(audio_files) > 10:
            st.warning("⚠️ Too many files selected. Please select 10 or fewer files.")
        else:
            if st.button("🔍 Compare Audio Files", type="primary"):
                features_list = []
                file_names = []
                progress_bar = st.progress(0)
                
                for i, audio_file in enumerate(audio_files):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(audio_file.getvalue())
                        temp_path = tmp_file.name
                    
                    try:
                        with st.spinner(f"Processing {audio_file.name}..."):
                            features, _, _ = analyzer.extract_features(temp_path)
                            features_list.append(features[0])
                            file_names.append(audio_file.name)
                        
                        progress_bar.progress((i + 1) / len(audio_files))
                    
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                # Calculate similarity matrix
                similarity_matrix = analyzer.compare_audio_files(features_list)
                
                # Create similarity heatmap
                fig = px.imshow(
                    similarity_matrix,
                    x=file_names,
                    y=file_names,
                    color_continuous_scale='RdYlBu_r',
                    title="Audio Similarity Matrix (Cosine Similarity)",
                    aspect="auto"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig)
                
                # Show similarity table
                st.subheader("📊 Similarity Scores")
                similarity_df = pd.DataFrame(
                    similarity_matrix,
                    index=file_names,
                    columns=file_names
                )
                st.dataframe(similarity_df.style.format("{:.3f}"))
                
                # Find most and least similar pairs
                st.subheader("🔍 Key Findings")
                mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
                masked_sim = similarity_matrix.copy()
                masked_sim[~mask] = np.nan
                
                max_idx = np.nanargmax(masked_sim)
                max_i, max_j = np.unravel_index(max_idx, masked_sim.shape)
                
                min_idx = np.nanargmin(masked_sim)
                min_i, min_j = np.unravel_index(min_idx, masked_sim.shape)
                
                st.write(f"🔗 **Most similar pair**: {file_names[max_i]} ↔ {file_names[max_j]} (similarity: {similarity_matrix[max_i, max_j]:.3f})")
                st.write(f"🔀 **Least similar pair**: {file_names[min_i]} ↔ {file_names[min_j]} (similarity: {similarity_matrix[min_i, min_j]:.3f})")

elif task == "Batch Audio Processing":
    st.header("⚡ Batch Audio Processing")
    st.write("Process multiple audio files and generate a comprehensive report.")
    
    audio_files = st.file_uploader(
        "Choose multiple audio files",
        type=['wav', 'mp3', 'flac', 'm4a'],
        accept_multiple_files=True,
        key="batch_processing"
    )
    
    if audio_files:
        st.write(f"Selected {len(audio_files)} files")
        
        analysis_options = st.multiselect(
            "Choose analysis types:",
            ["MERT Features", "Audio Properties", "Similarity Analysis"],
            default=["MERT Features", "Audio Properties"]
        )
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            results = {}
            progress_bar = st.progress(0)
            
            for i, audio_file in enumerate(audio_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    temp_path = tmp_file.name
                
                try:
                    file_results = {"filename": audio_file.name}
                    
                    if "MERT Features" in analysis_options or "Similarity Analysis" in analysis_options:
                        features, y, sr = analyzer.extract_features(temp_path)
                        file_results["features"] = features[0]
                        file_results["feature_norm"] = np.linalg.norm(features)
                    
                    if "Audio Properties" in analysis_options:
                        if "features" not in file_results:
                            y, sr = analyzer.load_audio(temp_path)
                        properties = analyzer.analyze_audio_properties(y, sr)
                        file_results["properties"] = properties
                    
                    results[audio_file.name] = file_results
                    progress_bar.progress((i + 1) / len(audio_files))
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            st.success("✅ Batch processing completed!")
            
            # Display results
            if "Audio Properties" in analysis_options:
                st.subheader("📊 Audio Properties Summary")
                props_data = []
                for filename, data in results.items():
                    if "properties" in data:
                        row = {"Filename": filename}
                        row.update(data["properties"])
                        props_data.append(row)
                
                if props_data:
                    props_df = pd.DataFrame(props_data)
                    st.dataframe(props_df)
                    
                    # Download option
                    csv = props_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Properties CSV",
                        data=csv,
                        file_name="audio_properties.csv",
                        mime="text/csv"
                    )
            
            if "MERT Features" in analysis_options:
                st.subheader("🔢 Feature Statistics")
                feature_stats = []
                for filename, data in results.items():
                    if "features" in data:
                        feature_stats.append({
                            "Filename": filename,
                            "Feature Norm": data["feature_norm"],
                            "Feature Mean": np.mean(data["features"]),
                            "Feature Std": np.std(data["features"])
                        })
                
                if feature_stats:
                    stats_df = pd.DataFrame(feature_stats)
                    st.dataframe(stats_df)
            
            if "Similarity Analysis" in analysis_options and len(audio_files) >= 2:
                st.subheader("🔄 Similarity Analysis")
                features_list = []
                file_names = []
                
                for filename, data in results.items():
                    if "features" in data:
                        features_list.append(data["features"])
                        file_names.append(filename)
                
                if len(features_list) >= 2:
                    similarity_matrix = analyzer.compare_audio_files(features_list)
                    
                    # Quick similarity heatmap
                    fig = px.imshow(
                        similarity_matrix,
                        x=file_names,
                        y=file_names,
                        color_continuous_scale='RdYlBu_r',
                        title="Batch Similarity Matrix"
                    )
                    st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About MERT
MERT (Music Understanding Model with Large-Scale Self-supervised Training) is a state-of-the-art model for music understanding tasks. It's trained on large-scale music data and can extract meaningful representations for various audio analysis tasks.

### Model Details
- **Model**: m-a-p/MERT-v1-330M
- **Parameters**: 330M
- **Sample Rate**: 24kHz
- **Applications**: Music similarity, genre classification, mood analysis, and more
""")

st.sidebar.markdown("""
### Features
- 🎵 Single audio analysis
- 🔢 Feature extraction
- 🔄 Audio comparison
- ⚡ Batch processing
- 📊 Comprehensive reporting
""")