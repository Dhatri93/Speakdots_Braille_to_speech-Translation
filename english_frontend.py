import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import webbrowser
from speech import text_to_speech

# INSTANT REDIRECT FUNCTION
def go_home():
    port = os.environ.get('STREAMLIT_SERVER_PORT', '8501')
    webbrowser.open_new_tab(f"http://localhost:{port}/app.py")
    st.stop()

# Set page config (must be first)
st.set_page_config(page_title="Braille Recognition", page_icon="✋", layout="wide")

# Load model and encoder - MODIFIED TO HANDLE IMPORT ERRORS
try:
    from segment import recognize_braille, load_model_and_encoder, preprocess_char_image
    @st.cache_resource
    def load_braille_model():
        try:
            model, label_encoder = load_model_and_encoder()
            return model, label_encoder
        except Exception as e:
            st.error(f"Model loading failed: {str(e)}")
            return None, None
except ImportError:
    st.error("Critical error: Could not import from segment.py")
    st.stop()

model, label_encoder = load_braille_model()

# App title and description
st.title("Braille Character Recognition")
st.markdown("""
Upload an image containing Braille characters, and this app will:
1. Segment the Braille dots
2. Recognize each Braille character
3. Convert it to English text
""")

# Sidebar for upload and settings
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Choose a Braille image", type=["jpg", "jpeg", "png"])
    
    st.subheader("Recognition Settings")
    min_confidence = st.slider("Minimum Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
    show_segmentation = st.checkbox("Show Segmentation Results", value=True)
    
    if st.button("🏠 Home"):
        go_home()

# Main content area
col1, col2 = st.columns(2)

if uploaded_file is not None and model is not None and label_encoder is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        results = recognize_braille(tmp_path, model, label_encoder, min_confidence)
        original_img = Image.open(uploaded_file)
        col1.image(original_img, caption="Original Image", use_column_width=True)
        col2.subheader("Recognition Results")
        col2.markdown(f"**English Text:** `{results['english']}`")
        
        # SPEECH BUTTON
        if st.button("🔊 Convert to Speech", key="eng_speech"):
            audio_bytes = text_to_speech(results['english'], 'en')
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
        
        if show_segmentation:
            st.subheader("Segmentation Visualization")
            vis_img = cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB)
            st.image(vis_img, caption="Segmented Characters", use_column_width=True)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
else:
    if uploaded_file is None:
        col1.image("https://via.placeholder.com/500x300?text=Upload+a+Braille+image", 
                  caption="No image uploaded yet", use_column_width=True)
        col2.markdown("""
        ### How to use this app:
        1. Upload a Braille image using the sidebar
        2. Adjust recognition settings if needed
        3. View the recognition results
        
        ### Tips for best results:
        - Use clear, well-lit images
        - Ensure Braille dots are distinct
        - Crop unnecessary background
        """)

if st.button("🏠 Return to Main Menu", key="bottom_home"):
    go_home()

st.caption("Braille Recognition App - Developed with Streamlit")