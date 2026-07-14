import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ============================================================
# 1. LOAD PHASE 1 MODEL (trained on Jigsaw)
# ============================================================
print("============================================================")
print("Loading Phase 1 model and tokenizer...")
model_path = r'C:\capstone\models\phase1'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# ============================================================
# 2. LOAD DATASET
# ============================================================
DATASET_FILE = r'C:\capstone\data\CN500-N500-T304.csv'
print(f"\nLoading dataset: {DATASET_FILE}")
df = pd.read_csv(DATASET_FILE)
df = df.dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

print(f"Total rows after cleaning: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print("============================================================")

# ============================================================
# 3. CLASS WEIGHTS
# ============================================================
print("\nCalculating class weights...")
labels_array = df['label'].values
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_array),
    y=labels_array
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Computed Weights: {weights_tensor}")
print("============================================================")

# ============================================================
# 4. TOKENIZE
# ============================================================
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label']
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset   = Dataset.from_pandas(val_df.reset_index(drop=True))

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# ============================================================
# 5. TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir=r'C:\capstone\models\phase2_temp',
    num_train_epochs=7,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    report_to='none',
)

# ============================================================
# 6. EVALUATION METRICS
# ============================================================
TARGET_NAMES = ['Constructive', 'Neutral', 'Toxic']

def compute_metrics(eval_pred):
    logits, eval_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(
        eval_labels, predictions,
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0
    )
    return {
        'f1_macro': report['macro avg']['f1-score'],
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
    }

# ============================================================
# 7. CUSTOM TRAINER WITH CLASS WEIGHTS
# ============================================================
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_batch = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels_batch.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ============================================================
# 8. TRAIN
# ============================================================
print("\nStarting Phase 2 training (500-500-304, Class Weights + Early Stopping)...")
trainer.train()

# ============================================================
# 9. FINAL EVALUATION
# ============================================================
print("\n============================================================")
print("Running final evaluation on best model...")
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
final_labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(
    final_labels, preds,
    target_names=TARGET_NAMES,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(final_labels, preds))
print("============================================================")

# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================
SAVE_PATH = r'C:\capstone\models\phase2_final'
print(f"\nSaving final model to {SAVE_PATH}...")
trainer.model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("Done! Model saved to models/phase2_final")
print("============================================================")