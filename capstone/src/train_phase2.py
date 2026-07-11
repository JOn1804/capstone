import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ============================================================
# LOAD PHASE 1 MODEL (trained on Jigsaw)
# ============================================================

print("Loading Phase 1 model and tokenizer...")
model_path = 'models/phase1'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# ============================================================
# LOAD HELLDIVERS 2 LABELED DATASET
# ============================================================

print("Loading Helldivers 2 labeled dataset...")
df = pd.read_csv('data/helldivers2_withoutnotes.csv')
df['label'] = df['label'].astype(int)

print(f"Total rows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# ============================================================
# TOKENIZE
# ============================================================

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir='models/phase2',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to='none',
)

# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(labels, predictions, target_names=['Neutral', 'Constructive', 'Toxic'], output_dict=True)
    return {
        'f1_macro': report['macro avg']['f1-score'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
    }

# ============================================================
# TRAIN
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting Phase 2 training...")
trainer.train()

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\nRunning final evaluation...")
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=['Neutral', 'Constructive', 'Toxic']))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, preds))

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving Phase 2 model...")
model.save_pretrained('models/phase2')
tokenizer.save_pretrained('models/phase2')
print("Done! Model saved to models/phase2")