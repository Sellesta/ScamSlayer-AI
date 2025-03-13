import torch
import tkinter as tk
from tkinter import font, ttk
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image, ImageTk

# Load the fine-tuned model
MODEL_PATH = "scam_slayer_final_model.pth"
TOKENIZER_PATH = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
model = BertForSequenceClassification.from_pretrained(TOKENIZER_PATH, num_labels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def classify_email():
    email_text = text_entry.get("1.0", tk.END).strip()
    if not email_text:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Please enter an email.")
        return
    
    inputs = tokenizer(email_text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]
    
    categories = ["Non-Malicious", "Malicious"]
    confidence = scores.max().item() * 100
    result = f"{categories[scores.argmax().item()]} ({confidence:.2f}%)"
    
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result)

# Initialize GUI
window = tk.Tk()
window.title("Scam Slayer")
window.geometry("800x600")
window.configure(bg="#FFFFFF")
custom_font = font.Font(family="Arial", size=12, weight="bold")

# Load logo
image_path = "logo.png"
original_image = Image.open(image_path)
resized_image = original_image.resize((180, 150), Image.ANTIALIAS)
tk_image = ImageTk.PhotoImage(resized_image)
image_label = tk.Label(window, image=tk_image, bg="#FFFFFF")
image_label.pack(pady=10)

# Label and text box
label = tk.Label(window, text="Enter your email text:", font=custom_font, bg="#FFFFFF")
label.pack(pady=10)
text_entry = tk.Text(window, height=10, width=65, font=custom_font, bg="#F0F0F0")
text_entry.pack(pady=10)

# Submit button
submit_button = ttk.Button(window, text="Analyze Email", command=classify_email)
submit_button.pack()

# Result box
result_text = tk.Text(window, height=5, width=65, font=custom_font, bg="#E6E6E6")
result_text.pack(pady=10)

window.mainloop()
