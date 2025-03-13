# Scam Slayer Wiki

## Overview
Scam Slayer is an AI-powered email threat detection system designed to identify and prevent phishing attacks. The model utilizes a fine-tuned BERT classifier trained on phishing email datasets to detect suspicious emails with high accuracy. With the increasing sophistication of phishing attempts and malicious websites, non-tenon-tech-savvy often find it challenging to differentiate between legitimate and suspicious content. ScamSlayer aims to provide an additional layer of security by proactively and contincontinuouslyages for potentially harmful popups and alerting users in real-time.

## Features
- **AI-Powered Email Threat Detection**
- **Fine-tuned BERT Model** for phishing classification
- **Real-time Email Screening**
- **Confidence Score for Threat Level**
- **API Integration for Business Use**

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Datasets
- Sklearn
- Google Colab (optional for training)

### Clone the Repository
```sh
# Clone Scam Slayer repository
git clone https://github.com/Sellesta/ScamSlayer-AI.git
cd ScamSlayer-AI
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Training the Model
### 1️⃣ Prepare Dataset
Ensure your phishing dataset is processed and available as `processed_merged_phishing_dataset.csv`.

### 2️⃣ Train the Model
```sh
python final_training.py
```
This script will train the BERT model with class balancing, early stopping, and evaluation metrics.

### 3️⃣ Save the Model
After training, save the final model:
```sh
trainer.save_model("scam_slayer_final_model")
```

## Deployment
### 1️⃣ Load the Trained Model
```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

model_path = "scam_slayer_final_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
```

### 2️⃣ Make Predictions
```python
def predict_email(email_text):
    inputs = tokenizer(email_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {"Safe": probs[0][0].item(), "Phishing": probs[0][1].item()}

# Test Example
email = "Your account has been compromised. Click here to reset your password."
print(predict_email(email))
```

## API Integration
You can deploy Scam Slayer as a REST API using **FastAPI**.

### 1️⃣ Install FastAPI
```sh
pip install fastapi uvicorn
```

### 2️⃣ Run API Server
Create a file `api.py`:
```python
from fastapi import FastAPI
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = FastAPI()
model_path = "scam_slayer_final_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_email(email_text):
    inputs = tokenizer(email_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {"Safe": probs[0][0].item(), "Phishing": probs[0][1].item()}

@app.post("/predict")
def classify_email(email: str):
    return predict_email(email)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run the API:
```sh
python api.py
```
Send a request using **Postman** or **curl**:
```sh
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"email": "Your account has been compromised."}'
```

## Future Improvements
- **Improve Model Accuracy**: Train on more diverse datasets.
- **Implement RNN-based Threat Detection**: Combine BERT with an RNN for sequential analysis.
- **Deploy on Cloud**: Host API on AWS/GCP.
- **Real-time Email Scanning**: Integrate with business email systems.

## Contributors
- **Moses Wanjema** - Data Engineer

## License
[MIT License](LICENSE)
