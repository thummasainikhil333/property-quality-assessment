"""
Property Quality Assessment System - ONNX Version
No OpenCV! No TensorFlow! Just lightweight ONNX Runtime
"""

import streamlit as st
import numpy as np
from PIL import Image
import json
import time

# ============================================
# IMPORTANT: Use ONNX Runtime, NOT TensorFlow!
# ============================================
import onnxruntime as ort

# Page configuration
st.set_page_config(
    page_title="Property Quality Assessment",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A5F;
        text-align: center;
    }
    .good-box {
        background-color: #90EE90;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .minor-box {
        background-color: #FFD700;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .major-box {
        background-color: #FF6B6B;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD ONNX MODEL (Cached for speed)
# ============================================

@st.cache_resource
def load_onnx_model():
    """Load ONNX model and create inference session"""
    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Create ONNX Runtime session
    # The file name must match what you upload to GitHub
    sess = ort.InferenceSession('property_quality_model.onnx', providers=['CPUExecutionProvider'])
    
    return sess, class_names

# ============================================
# PREPROCESSING (No OpenCV! Just PIL + NumPy)
# ============================================

def preprocess_image(image):
    """
    Preprocess image for ONNX model input
    No OpenCV needed - pure PIL and NumPy
    """
    # Resize to 224x224 (MobileNetV2 input size)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

def calculate_quality_score(predictions, class_names):
    """Calculate 0-100 quality score"""
    weights = {'good': 100, 'minor': 50, 'major': 0}
    score = 0
    for i, class_name in enumerate(class_names):
        score += predictions[0][i] * weights.get(class_name.lower(), 0)
    return score

def get_recommendation(quality_score, predicted_class):
    """Generate business recommendation"""
    if quality_score >= 70:
        return "**RECOMMENDATION:** Proceed with standard purchase. Schedule professional inspection for documentation. Estimated repairs: $0-$5,000"
    elif quality_score >= 40:
        return " **RECOMMENDATION:** Conduct focused inspection on defect areas. Factor repair costs ($5,000-$15,000) into offer price."
    else:
        return " **RECOMMENDATION:** Proceed with caution. Major renovation likely required ($25,000+). Consult structural engineer before making an offer."

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-title">🏠 Property Quality Assessment System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("###  How It Works")
        st.markdown("""
        1. Upload a clear property facade photo
        2. Our AI analyzes the image
        3. Get instant assessment
        
        **Quality Classes:**
        -  **Good:** Well-maintained
        -  **Minor:** Peeling paint, stains
        -  **Major:** Cracks, structural damage
        """)
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown("""
        - **Architecture:** MobileNetV2 (ONNX)
        - **Accuracy:** 99%
        - **Inference:** <300ms
        - **No OpenCV needed!**
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Property Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the building's front facade"
        )
    
    # Load model
    with st.spinner("Loading AI model (lightweight ONNX)..."):
        sess, class_names = load_onnx_model()
    
    # Process uploaded image
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        with col2:
            st.markdown("### Uploaded Property")
            st.image(image, use_column_width=True)
        
        # Predict
        with st.spinner("Analyzing property condition..."):
            start_time = time.time()
            
            # Preprocess
            processed_img = preprocess_image(image)
            
            # Run inference with ONNX
            input_name = sess.get_inputs()[0].name
            predictions = sess.run(None, {input_name: processed_img})[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            predicted_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
            quality_score = calculate_quality_score(predictions, class_names)
        
        # Display results
        st.markdown("---")
        st.markdown("### Assessment Results")
        
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predicted Class", predicted_class.upper())
        with m2:
            st.metric("Confidence", f"{confidence:.1f}%")
        with m3:
            st.metric("Quality Score", f"{quality_score:.0f}/100")
        with m4:
            st.metric("Inference Time", f"{inference_time:.0f}ms")
        
        # Color-coded result box
        if predicted_class.lower() == 'good':
            st.markdown(f'<div class="good-box"><h2> GOOD CONDITION</h2><p>Confidence: {confidence:.1f}%</p></div>', unsafe_allow_html=True)
        elif predicted_class.lower() == 'minor':
            st.markdown(f'<div class="minor-box"><h2> MINOR DEFECTS</h2><p>Confidence: {confidence:.1f}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="major-box"><h2> MAJOR DAMAGE</h2><p>Confidence: {confidence:.1f}%</p></div>', unsafe_allow_html=True)
        
        # Recommendation
        st.markdown("###  Investment Recommendation")
        recommendation = get_recommendation(quality_score, predicted_class)
        
        if quality_score >= 70:
            st.success(recommendation)
        elif quality_score >= 40:
            st.warning(recommendation)
        else:
            st.error(recommendation)
        
        # Probability breakdown
        with st.expander(" View Detailed Probabilities"):
            for i, class_name in enumerate(class_names):
                prob = predictions[0][i] * 100
                st.write(f"- **{class_name.upper()}:** {prob:.1f}%")
                st.progress(int(prob))

# ============================================
# RUN THE APP
# ============================================

if __name__ == "__main__":
    main()
