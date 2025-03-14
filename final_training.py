import os
os.environ["WANDB_DISABLED"] = "true"

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

# ✅ Load dataset
dataset = load_dataset("csv", data_files={"train": "processed_merged_phishing_dataset.csv"})
df = dataset['train'].to_pandas()  # Convert to pandas for easier debugging

# ✅ Ensure the "text" column exists
if "text" not in df.columns:
    raise KeyError("Error: The dataset is missing the 'text' column!")

# ✅ Convert None values to empty strings
df["text"] = df["text"].astype(str).fillna("")

# ✅ Split dataset into training (80%) and validation (20%)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# ✅ Convert back to Hugging Face datasets
train_dataset = DatasetDict({"train": dataset["train"].from_pandas(train_df)})
val_dataset = DatasetDict({"validation": dataset["train"].from_pandas(val_df)})

# ✅ Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Tokenize text safely
def tokenize_function(examples):
    if "text" not in examples:
        raise ValueError("Error: 'text' column missing during tokenization!")
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# ✅ Ensure correct column names
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

# ✅ Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ✅ Compute Class Weights
labels_np = np.array(train_df["label"])  # Convert to numpy array
class_weights = compute_class_weight('balanced', classes=np.unique(labels_np), y=labels_np)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# ✅ Define Model & Loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Use weighted loss to handle imbalanced data
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# ✅ Define Training Arguments (Now with Evaluation Strategy)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # 🔥 Now valid because we have `eval_dataset`
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=1e-5,  # 🔥 Lower learning rate for better fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,  # 🔥 More epochs for deeper learning
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10
)

# ✅ Define Metrics for Evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=1),
        "recall": recall_score(labels, predictions, zero_division=1),
        "f1": f1_score(labels, predictions, zero_division=1)
    }

# ✅ Trainer (Now with Validation Data)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=val_dataset["validation"],  # ✅ Now we have an evaluation dataset
    compute_metrics=compute_metrics
)

# ✅ Train Model
trainer.train()

# ✅ Save Model
trainer.save_model("scam_slayer_final_model")
