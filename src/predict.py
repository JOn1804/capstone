import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Point this at the FOLDER, not any individual file inside it
MODEL_DIR = 'models/phase1'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

label_names = {0: "Neutral", 1: "Constructive Negative", 2: "Toxic"}

def predict(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_names[pred_id]

samples = [
    "This game is amazing, I love the new update!",
    "I think the balancing of weapons needs work, but I still enjoy playing.",
    "This game is garbage and the devs are idiots who don't know what they're doing.",
]

for s in samples:
    print(f"\nText: {s}")
    print(f"Prediction: {predict(s)}")