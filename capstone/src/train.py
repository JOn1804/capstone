import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ============================================================
# DEVICE CHECK
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD AND PREPARE JIGSAW DATASET (FULL DATASET NOW)
# ============================================================
print("\nLoading full Jigsaw dataset...")
df = pd.read_csv('data/train.csv')
print(f"Total rows: {len(df)}")

toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_label(row):
    if row[toxic_cols].sum() > 0:
        return 2  # Toxic
    return 0  # Neutral

df['label'] = df.apply(get_label, axis=1)
df = df[['comment_text', 'label']].rename(columns={'comment_text': 'text'})

# ============================================================
# TOKENIZE
# ============================================================
print("\nLoading DistilBERT tokenizer (fast/Rust-backed)...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ============================================================
# LOAD MODEL
# ============================================================
print("\nLoading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)

# ============================================================
# TRAINING ARGUMENTS (GPU-tuned)
# ============================================================
training_args = TrainingArguments(
    output_dir='models/phase1',
    num_train_epochs=3,
    per_device_train_batch_size=32,   # bumped up since GPU has headroom
    per_device_eval_batch_size=32,
    eval_strategy='epoch',
    save_strategy='epoch',
    report_to='none',
    logging_steps=100,
    load_best_model_at_end=True,
    fp16=(device == "cuda"),          # mixed precision -> big speedup on GPU, skipped on CPU
)

# ============================================================
# TRAIN
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\nStarting Phase 1 training (full dataset)...")
trainer.train()

print("\nSaving model...")
model.save_pretrained('models/phase1')
tokenizer.save_pretrained('models/phase1')
print("Done! Model saved to models/phase1")