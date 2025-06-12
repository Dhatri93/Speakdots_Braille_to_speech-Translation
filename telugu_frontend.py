import streamlit as st
from telugu import BrailleToTelugu
import os
import webbrowser
from speech import text_to_speech

# Set page config (must be first)
st.set_page_config(
    page_title="Braille to Telugu Converter", 
    page_icon=":pencil2:",
    layout="centered"
)

# INSTANT REDIRECT
def navigate_to(page):
    port = os.environ.get('STREAMLIT_SERVER_PORT', '8501')
    url = f"http://localhost:{port}/{page}"
    webbrowser.open_new_tab(url)
    st.stop()

# Your converter UI
st.title("Braille to Telugu Converter")
converter = BrailleToTelugu()

braille_input = st.text_area("Enter Braille Text:", height=150)
if st.button("Convert to Telugu"):
    if braille_input.strip():
        try:
            telugu_output = converter.convert_to_telugu(braille_input)
            st.text_area("Telugu Output:", value=telugu_output, height=150)
            st.success("Conversion successful!")
            
            # Store the output in session state
            st.session_state.telugu_output = telugu_output
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# SPEECH BUTTON (only appears after conversion)
if 'telugu_output' in st.session_state:
    if st.button("🔊 వినండి (Listen)"):
        audio_bytes = text_to_speech(st.session_state.telugu_output, 'te')
        if audio_bytes:
            st.audio(audio_bytes, format='audio/mp3')

# Home button
st.markdown("---")
if st.button("🏠 Home"):
    navigate_to("app.py")

if st.button("Reset"):
    st.session_state.clear()
    st.rerun()