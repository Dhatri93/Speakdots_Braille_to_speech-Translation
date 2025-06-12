from gtts import gTTS
from io import BytesIO
import streamlit as st

def text_to_speech(text, language='en'):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang=language)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        return audio_bytes
    except Exception as e:
        st.error(f"Speech synthesis failed: {str(e)}")
        return None