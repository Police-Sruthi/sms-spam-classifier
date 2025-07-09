


import streamlit as st
from transformers import pipeline

# Load BERT-based text classification model from Hugging Face
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

# Set page
st.set_page_config(page_title="SMS Spam Classifier (BERT)", layout="centered")
st.title("🤖 SMS Spam Classifier using BERT")
st.markdown("Enter any message below and check if it's spam or not — powered by a deep learning model.")

# Load model
classifier = load_model()

# User input
user_input = st.text_area("📩 Type your message")

if st.button("Detect"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        result = classifier(user_input)[0]
        label = result['label']
        score = round(result['score'] * 100, 2)
        
        if label == "spam":
            st.error(f"🚫 Spam detected ({score}%)")
        else:
            st.success(f"✅ Not Spam ({score}%)")
