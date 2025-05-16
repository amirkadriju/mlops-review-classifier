import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer from checkpoint
model_path = './best_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Test input
texts = [
    "This product is amazing!",
    "Worst experience ever. Not recommended.",
    "very good.",
    "not."
]

# Tokenize inputs
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=300)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

# Convert predictions to labels
id2label = model.config.id2label
predicted_labels = [id2label[int(pred)] for pred in predictions]

# Print results
for text, label in zip(texts, predicted_labels):
    print(f"Text: {text}\nPredicted label: {label}\n")
