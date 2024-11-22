# Importare le librerie necessarie
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# 1. Caricare i dati
# Utilizzare la variabile d'ambiente per il percorso dei dati di input
file_path = os.path.join('./data/input/vgsales.csv')
df = pd.read_csv(file_path)

# Visualizzare le prime righe per verificare il caricamento corretto
print(df.head())

# 2. Pre-processing
# Selezionare solo le colonne rilevanti per l'analisi
# Manteniamo "Name", "Genre", "Platform", "Year", "Publisher" per il nostro scopo
df = df[['Name', 'Genre', 'Platform', 'Year', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]

# Rimuovere righe con valori nulli nelle colonne critiche
df = df.dropna(subset=['Genre', 'Platform', 'Year', 'Publisher'])

# Codificare la variabile target `Genre` (il genere)
df['Genre_encoded'] = LabelEncoder().fit_transform(df['Genre'])

# Convertire il testo in numeri usando TfidfVectorizer per la colonna 'Name' (titolo del gioco)
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
name_matrix = vectorizer.fit_transform(df['Name'])

# Codificare le variabili categoriche come 'Platform' e 'Publisher'
df['Platform_encoded'] = LabelEncoder().fit_transform(df['Platform'])
df['Publisher_encoded'] = LabelEncoder().fit_transform(df['Publisher'])

# Combinare tutte le caratteristiche (aggiungendo le vendite per regione e anno)
X = np.hstack((
    name_matrix.toarray(), 
    df[['Platform_encoded', 'Publisher_encoded', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].values
))

# Variabile target
y = df['Genre_encoded']

# 3. Suddivisione in train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Addestramento del modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Valutazione del modello
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Salva il modello addestrato nella directory del modello di SageMaker
joblib.dump(model, 'vgsales_model.pkl')

# Salvataggio del vettorizzatore per l'uso futuro
joblib.dump(vectorizer, 'name_vectorizer.pkl')
