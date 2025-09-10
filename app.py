import streamlit as st
from transformers import pipeline
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the model once and cache it"""
    model_id = "krishanmittal018/distilhubert-finetuned-gtzan"
    return pipeline("audio-classification", model=model_id)

def classify_audio(audio_file):
    """Classify the audio file and return predictions"""
    if audio_file is None:
        return None
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(audio_file.getvalue())
        temp_path = tmp_file.name
        
        try:
            # Get predictions
            pipe = load_model()
            preds = pipe(temp_path)
            return preds
        finally:
            # Clean up temp file
            os.remove(temp_path)

def main():
    st.title("🎵 Music Genre Classification")
    st.markdown("Upload an audio file to classify its genre")
    
    # File uploader
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
    
    if audio_file:
        # Display audio player
        st.audio(audio_file)
        
        with st.spinner('Analyzing audio...'):
            predictions = classify_audio(audio_file)
            
            if predictions:
                # Create columns for visualization
                st.subheader("Genre Predictions")
                
                # Convert predictions to more readable format
                for pred in predictions:
                    confidence = pred['score'] * 100
                    st.progress(confidence/100)
                    st.write(f"{pred['label']}: {confidence:.1f}%")

if __name__ == "__main__":
    main()