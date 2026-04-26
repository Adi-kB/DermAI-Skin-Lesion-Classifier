import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="DermAI", layout="wide", initial_sidebar_state="collapsed")

# --- MASTER CSS (Merging Stable Home + Advanced Analysis) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&family=Montserrat:wght@400;700&display=swap');
    /* Global Background */
    .stApp {
        background: linear-gradient(180deg, #f8f4ff 0%, #ffffff 100%) !important;
        font-family: 'Montserrat', sans-serif;
    }
    /* 1. STABLE HOME PAGE LAYOUT (From your second code) */
    .hero-container {
        position: fixed;
        top: 35%; 
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        width: 100%;
        z-index: 999;
    }
    .logo-main {
        font-family: 'Dancing Script', cursive !important;
        color: #4a148c !important;
        font-size: 130px !important;
        margin-bottom: 20px !important;
        display: block;
    }
    /* 2. ANALYSIS PAGE CALLIGRAPHY HEADER */
    .analysis-header {
        font-family: 'Dancing Script', cursive !important;
        color: #311b92 !important;
        font-size: 85px !important;
        text-align: center;
        margin-top: -30px !important;
        padding-bottom: 10px;
    }
    /* 3. COMPLIMENTARY CONTRAST INSTRUCTIONS */
    .instruction-note {
        background-color: #ede7f6;
        color: #4527a0;
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 30px;
        border: 1px solid #d1c4e9;
    }
    /* 4. BUTTON STYLING (The centered version from first code) */
    div.stButton > button {
        background-color: #6a1b9a !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 18px 80px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border: none !important;
        margin-left: 25px !important; 
        box-shadow: 0 10px 25px rgba(106, 27, 154, 0.3) !important;
        transition: 0.3s !important;
    }
    /* 5. RESULT TYPOGRAPHY & VISIBLE INFO BOX (First Code Styles) */
    .lesion-title { color: #311b92; font-size: 65px; font-weight: 800; line-height: 1; margin-bottom: 10px; }
    .conf-text { color: #7b1fa2; font-size: 28px; font-weight: 600; margin-bottom: 25px; }
    .disclaimer-box { 
        background: #fff9c4; 
        border-left: 10px solid #fbc02d; 
        padding: 20px; 
        border-radius: 10px; 
        color: #333; 
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    .info-box { 
        background: #f3e5f5; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #e1bee7; 
        color: #2e004f !important; 
        font-weight: 600;
    }
    #MainMenu, footer, header {visibility: hidden !important;}
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('ham10000_model_full.h5')

model = load_my_model()
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
lesion_data = {
    'akiec': {"name": "Actinic Keratosis", "info": "Precancerous scaly patches from sun damage."},
    'bcc': {"name": "Basal Cell Carcinoma", "info": "Common, slow-growing skin cancer."},
    'bkl': {"name": "Benign Keratosis", "info": "Common non-cancerous skin growths."},
    'df': {"name": "Dermatofibroma", "info": "Benign firm bumps often on the legs."},
    'mel': {"name": "Melanoma", "info": "Serious skin cancer—requires urgent professional review."},
    'nv': {"name": "Melanocytic Nevi", "info": "Typical moles. Monitor for changes in shape/color."},
    'vasc': {"name": "Vascular Lesion", "info": "Benign lesions related to blood vessels."}
}

# --- NAVIGATION ---
if 'pg' not in st.session_state:
    st.session_state.pg = 'home'

# --- HOME PAGE (Using the layout from your second code) ---
if st.session_state.pg == 'home':
    st.markdown('<div class="hero-container"><div class="logo-main">DermAI</div></div>', unsafe_allow_html=True)
    
    # Exact spacer you used in the second code to position the button
    st.write("<div style='height: 42vh;'></div>", unsafe_allow_html=True)
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        if st.button("Begin Analysis"):
            st.session_state.pg = 'app'
            st.rerun()

# --- ANALYSIS PAGE (Exactly as it was in your first code) ---
else:
    if st.button("← Back Home"):
        st.session_state.pg = 'home'
        st.rerun()
            
    st.markdown('<div class="analysis-header">DermAI Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="instruction-note">Please upload a clear, high-resolution image of the skin lesion for neural evaluation.</div>', unsafe_allow_html=True)
    
    file = st.file_uploader("", type=["jpg","png","jpeg"])
    
    if file:
        col_left, col_right = st.columns([1.5, 1], gap="large")
        img = Image.open(file).convert("RGB")
        
        with col_left:
            st.image(img, use_container_width=True)
        
        with col_right:
            prep = img.resize((224, 224))
            arr = np.array(prep) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            p = model.predict(arr)
            idx = np.argmax(p)
            d = lesion_data.get(class_names[idx])
            
            st.markdown(f"<div class='lesion-title'>{d['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='conf-text'>AI Confidence: {np.max(p)*100:.2f}%</div>", unsafe_allow_html=True)
            
            st.markdown("""
                <div class="disclaimer-box">
                    ⚠️ DISCLAIMER: This is an AI research tool. 
                    Results must be verified by a professional dermatologist.
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="info-box">
                    <strong>About this lesion:</strong><br>
                    {d['info']}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("New Scan"):
                st.rerun()