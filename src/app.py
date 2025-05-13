from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
model_path = './best_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label

def classify_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=300)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1)
    return id2label[int(prediction)]

@app.get("/", response_class=HTMLResponse)
async def form_get():
    return """
    <html>
        <body>
            <h2>Review Classifier</h2>
            <form action="/predict" method="post">
                <textarea name="review" rows="4" cols="50"></textarea><br><br>
                <input type="submit" value="Classify">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def form_post(review: str = Form(...)):
    label = classify_text(review)
    return f"""
    <html>
        <body>
            <h2>Prediction Result</h2>
            <p><strong>Review:</strong> {review}</p>
            <p><strong>Predicted Label:</strong> {label}</p>
            <a href="/">Try another</a>
        </body>
    </html>
    """
