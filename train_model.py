from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# === Load merged dataset ===
df = pd.read_csv("merged_dataset.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# === Tokenization ===
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

# === Convert to Hugging Face Dataset format ===
train_dataset = Dataset.from_dict({**train_encodings, "label": train_labels.tolist()})
val_dataset = Dataset.from_dict({**val_encodings, "label": val_labels.tolist()})

# === Load pre-trained multilingual BERT model ===
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# === Training Configuration ===
training_args = TrainingArguments(
    output_dir="saved_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1
)

# === Evaluation Metric ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), axis=1)
    accuracy = (preds == torch.tensor(labels)).float().mean().item()
    return {"accuracy": accuracy}

# === Trainer Setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# === Train and Save ===
trainer.train()
trainer.save_model("saved_model")
tokenizer.save_pretrained("saved_model")
print("✅ Training complete. Model saved in 'saved_model/'")
