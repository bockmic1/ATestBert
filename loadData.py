# text_classifier.py

from datasets import load_dataset

# Laden des Datensatzes
dataset = load_dataset("emotion")

# Datensatz in Trainings- und Validierungssätze aufteilen
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# Beispiel eines Datensatzes anzeigen
print(train_dataset[0])
