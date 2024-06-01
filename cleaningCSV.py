import pandas as pd
import csv

# CSV-Datei einlesen und fehlerhafte Zeilen überspringen
cleaned_rows = []
with open('dataSet.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) == 2:
            cleaned_rows.append(row)

# Bereinigte CSV-Datei speichern
with open('cleaned_dataSet.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(cleaned_rows)
