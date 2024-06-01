import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# CSV-Datei einlesen
df = pd.read_csv('cleaned_dataSet.csv', encoding='utf-8')

# Labels bereinigen und in numerische Werte umwandeln
label_mapping = {'Real': 0, 'Fake': 1}
df['label'] = df['label'].map(label_mapping)

# Überprüfen Sie, ob es keine None-Werte gibt
df = df.dropna(subset=['label'])

# Überprüfen Sie, ob die CSV-Datei korrekt eingelesen wurde
print(df.head())

# Konvertieren in Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Aufteilen in Trainings- und Validierungsdatensätze
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
val_dataset = dataset['test']

# Tokenizer laden und Daten tokenisieren
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(example["content"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Modell erstellen
num_labels = len(label_mapping)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

# Trainingsparameter festlegen
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Funktionen zur Berechnung der Metriken
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="weighted")
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Modell trainieren
trainer.train()

# Modell bewerten
eval_result = trainer.evaluate()
print(f"Evaluation results: {eval_result}")

# Vorhersagen treffen
def predict(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

# Beispielvorhersagen
texts = ["This is a new vaccine for Alzheimer.", "The cure for cancer has been found."]
predictions = predict(texts)
print(predictions)
