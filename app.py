import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Blood Group Detection",
    page_icon="🩸",
    layout="centered"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_blood_group_model():
    """Load the trained model with caching"""
    try:
        # Changed from hardcoded Windows path to relative path
        model_path = "bloodgroup_cnn_model.h5"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Blood group labels
class_labels = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# App title
st.markdown('<h1 class="main-header">🩸 Fingerprint Blood Group Detection</h1>', unsafe_allow_html=True)

# Load model
model = load_blood_group_model()

if model is not None:
    # File upload section
    st.subheader("Upload Fingerprint Image")
    uploaded_file = st.file_uploader(
        "Choose a fingerprint image", 
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, BMP"
    )

    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
            st.subheader("Uploaded Image")
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Fingerprint', use_column_width=True)

        with col2:
            # Preprocess and predict
            st.subheader("Analysis Results")
            
            try:
                # Preprocess image
                img_processed = img.convert('L')  # convert to grayscale
                img_processed = img_processed.resize((128, 128))
                x = image.img_to_array(img_processed) / 255.0
                x = np.expand_dims(x, axis=0)

                # Make prediction
                with st.spinner('Analyzing fingerprint...'):
                    pred = model.predict(x)
                    pred_class = np.argmax(pred)
                    confidence = np.max(pred) * 100

                # Display results
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                
                # Blood group with emoji
                blood_group_emojis = {
                    'A+': '🅰️➕', 'A-': '🅰️➖',
                    'B+': '🅱️➕', 'B-': '🅱️➖', 
                    'AB+': '🆎➕', 'AB-': '🆎➖',
                    'O+': '🅾️➕', 'O-': '🅾️➖'
                }
                
                emoji = blood_group_emojis.get(class_labels[pred_class], '🩸')
                
                st.success(f"**Predicted Blood Group:** {emoji} {class_labels[pred_class]}")
                
                # Confidence with color coding
                if confidence > 80:
                    st.info(f"**Confidence:** {confidence:.2f}% ✅")
                elif confidence > 60:
                    st.warning(f"**Confidence:** {confidence:.2f}% ⚠️")
                else:
                    st.error(f"**Confidence:** {confidence:.2f}% ❗")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show all probabilities
                st.subheader("Detailed Probabilities")
                for i, prob in enumerate(pred[0]):
                    progress_bar = st.progress(float(prob))
                    st.write(f"{class_labels[i]}: {prob*100:.2f}%")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    # Add information section
    with st.expander("ℹ️ About this App"):
        st.write("""
        This application uses a Convolutional Neural Network (CNN) to predict blood groups from fingerprint images.
        
        **How it works:**
        1. Upload a fingerprint image (PNG, JPG, or JPEG)
        2. The image is preprocessed and analyzed by the AI model
        3. The model predicts the blood group with confidence percentage
        
        **Note:** This is for educational/demonstration purposes. Always verify blood groups through medical tests.
        """)

else:
    st.error("Unable to load the model. Please check the model file path and try again.")

# Add footer
st.markdown("---")
st.markdown("**Disclaimer:** This tool is for demonstration purposes only. Always consult medical professionals for accurate blood group testing.")
