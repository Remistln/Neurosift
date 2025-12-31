import streamlit as st
import os
import sys
import pandas as pd
import json
from sqlalchemy.orm import sessionmaker

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import LOCAL_DATA_DIR
from src.collector.metadata_store import MetadataStore, ImageMetadata
from src.inference.predictor import ModelPredictor

# Initialize AI Predictor
@st.cache_resource
def get_predictor():
    return ModelPredictor()

predictor = get_predictor()

# Page Config
st.set_page_config(page_title="NeuroSift DICOM Viewer", layout="wide")

# Title
st.title("NeuroSift: DICOM Viewer")

# Connect to DB
store = MetadataStore()
Session = sessionmaker(bind=store.engine)
session = Session()

# Load test patients
try:
    with open("data/splits.json", "r") as f:
        splits = json.load(f)
    test_patients = splits.get("test", [])
except FileNotFoundError:
    st.error("splits.json not found")
    test_patients = []

# Initialize Session State for predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Tabs
tab_gallery, tab_upload = st.tabs(["Gallery", "Upload Test"])

with tab_gallery:
    # Sidebar
    st.sidebar.header("Cohort Navigator")
    
    # Filter test patients
    if not test_patients:
         st.error("No test patients found")
    else:
        selected_patient = st.sidebar.selectbox("Select Patient", test_patients)
        
        # Filter images (Anatomical only)
        target_mods = ["T1", "T2", "FLAIR"]
        query = session.query(ImageMetadata).filter(
            ImageMetadata.pmc_id == selected_patient,
            ImageMetadata.modality.in_(target_mods)
        )
        images = query.all()
        
        st.sidebar.metric("Slices for Patient", len(images))
        
        # Analyze All Button
        if st.sidebar.button("Analyze All Images"):
            progress_bar = st.sidebar.progress(0)
            for idx, img in enumerate(images):
                if os.path.exists(img.s3_key):
                    res = predictor.predict(img.s3_key)
                    if res:
                        st.session_state.predictions[img.id] = res
                progress_bar.progress((idx + 1) / len(images))
            st.sidebar.success("Analysis Complete")
        
        if images:
             st.sidebar.markdown(f"**Metadata**:\n{images[0].caption}")
        
        st.header(f"Patient: {selected_patient}")
        st.info("Displaying test set images only")
    
        # Grid
        cols = st.columns(3)
        for idx, img in enumerate(images):
            col = cols[idx % 3]
            with col:
                if os.path.exists(img.s3_key):
                    st.image(img.s3_key, caption=f"Slice {idx+1}")
                    
                    # Individual Button
                    btn_key = f"btn_{img.id}"
                    
                    # Check if we have a prediction in session state
                    pred = st.session_state.predictions.get(img.id)
                    
                    if not pred and st.button("Analyze", key=btn_key):
                        pred = predictor.predict(img.s3_key)
                        st.session_state.predictions[img.id] = pred
                        st.rerun() # Refresh to show result
                        
                    if pred:
                        conf = pred['confidence'] * 100
                        color = "green" if conf > 90 else "orange"
                        st.markdown(f"**Prediction**: :{color}[{pred['label']}] ({conf:.1f}%)")
                        
                        if img.modality and img.modality != "Unknown":
                            if img.modality == pred['label']:
                                st.caption("Matches Ground Truth")
                            else:
                                st.caption(f"GT: {img.modality}")

with tab_upload:
    st.header("Modality Test")
    st.markdown("Upload external MRI for validation")
    
    uploaded_file = st.file_uploader("Choose DICOM or Image", type=['png', 'jpg', 'jpeg', 'dcm'])
    
    if uploaded_file is not None:
        # Save temp file
        with open("temp_upload.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns(2)
        with col1:
             st.image("temp_upload.png", caption="Uploaded Image", width=300)
        
        with col2:
            st.markdown("### Diagnosis")
            if st.button("Analyze Upload"):
                res = predictor.predict("temp_upload.png")
                if res:
                     st.metric("Predicted Modality", res['label'])
                     st.progress(res['confidence'])
                     st.caption(f"Confidence: {res['confidence']*100:.1f}%")

session.close()
