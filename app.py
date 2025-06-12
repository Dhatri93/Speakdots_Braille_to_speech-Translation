import streamlit as st
import os

def navigate_to(page):
    st.session_state.page = page
    raise st.runtime.scriptrunner.StopException

def main():
    # Initialize session state
    if 'language' not in st.session_state:
        st.session_state.language = None
    
    # Set page config (must be first Streamlit command)
    st.set_page_config(
        page_title="Speak Dots - Braille Translator",
        page_icon="🖐️",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .title {
            text-align: center;
            color: #2b5876;
            font-size: 3rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #2b5876, #4e4376);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            text-align: center;
            color: #4e4376;
            font-size: 1.2rem;
            margin-bottom: 2.5rem;
        }
        
        .stButton>button {
            width: 250px;
            height: 120px;
            border-radius: 15px;
            font-size: 1.5rem !important;
            font-weight: 600;
            margin: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border: none;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #4e4376;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        }
        
        .button-container {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 2rem;
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #7a7a7a;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # App title and subtitle
    st.markdown("<h1 class='title'>Speak Dots</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Braille Translation System</p>", unsafe_allow_html=True)
    
    # Button container
    st.markdown("<div class='button-container'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("English", key="english"):
            st.session_state.language = "English"
            st.success("English mode selected")
            navigate_to("english_frontend.py")
    
    with col2:
        if st.button("Telugu", key="telugu"):
            st.session_state.language = "Telugu"
            st.success("Telugu mode selected")
            navigate_to("telugu_frontend.py")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Empowering communication through tactile language</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if 'page' in st.session_state:
        os.system(f"streamlit run {st.session_state.page}")
    else:
        main()




