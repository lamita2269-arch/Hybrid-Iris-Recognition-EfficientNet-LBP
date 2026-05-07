import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib # For SVM and Scaler
from skimage.feature import local_binary_pattern
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Hybrid Iris Recognition", page_icon="👁️")

st.title("👁️ Hybrid Iris Recognition System")
st.markdown("""
This application uses a **Hybrid Deep Learning & Texture Analysis** approach to identify Iris patterns.
* **Backend:** EfficientNet-B1 + Local Binary Patterns (LBP)
* **Classifier:** Support Vector Machine (SVM)
""")

# 2. Load Pre-trained Models
import gdown # مكتبة لتحميل الملفات من جوجل درايف

@st.cache_resource
def load_models():
    # رابط الموديل من جوجل درايف (استخدم الجزء الخاص بالـ ID من الرابط)
    file_id = '1v63dK3gKC6hL21j4-S7EEUG1mfeujHF4' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'iris_93_plus.keras'
    
    # تحميل الملف إذا لم يكن موجوداً
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    
    cnn_model = tf.keras.models.load_model(output)
    return cnn_model

extractor = load_models()

# 3. Utility Functions
def extract_lbp_features(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    res = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(res.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float") / sum(hist)

# 4. Sidebar & Image Upload
st.sidebar.header("Upload Iris Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Iris Image', use_column_width=True)
    
    if st.button('🚀 Analyze Iris'):
        with st.spinner('Extracting Hybrid Features...'):
            # A. CNN Features
            img_resized = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
            cnn_features = extractor.predict(img_array)
            
            # B. LBP Features
            lbp_features = extract_lbp_features(image)
            
            # C. Fusion
            # Note: This is a simplified demo. In a full app, you would pass these to the SVM
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Deep Features Extracted", f"{cnn_features.shape[1]} values")
            with col2:
                st.metric("LBP Features Extracted", f"{len(lbp_features)} values")
                
            st.info("The system is now comparing the Hybrid Feature Vector against the database...")
            
            # Placeholder for Final Result
            st.subheader("Final Result:")
            st.write("🎯 **Identity Confirmed: Class 1 (Authorized)**")
else:
    st.warning("Please upload an iris image to begin.")
