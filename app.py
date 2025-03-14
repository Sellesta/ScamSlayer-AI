import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model from Hugging Face
MODEL_NAME = "sellestas/scam_slayer_model"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def classify_email(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, prediction = torch.max(probabilities, dim=-1)
    
    label_map = {0: "Non-Malicious", 1: "Malicious"}
    return label_map[prediction.item()], confidence.item() * 100

# Streamlit UI
st.set_page_config(page_title="Scam Slayer", layout="centered")
st.image("logo.png", width=150)
st.title("🛡️ Scam Slayer - AI Email Threat Detector")
st.markdown("### 🔍 Detect phishing and malicious emails instantly!")

email_text = st.text_area("✉️ Paste the email content below:", height=200)

if st.button("Detect Scam", help="Click to analyze the email content"):
    if email_text.strip():
        category, confidence = classify_email(email_text)
        st.success(f"**🔹 Result: {category} ({confidence:.2f}% Confidence)**")
        st.markdown("\n ✅ Stay vigilant! 🚀")
    else:
        st.warning("⚠️ Please enter email content to analyze!")
