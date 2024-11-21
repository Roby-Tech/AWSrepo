import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Funzione per elencare i file e le directory
def list_files_and_directories(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

# Percorsi configurati dalle variabili d'ambiente
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')  # Dove salvare il modello
output_dir = os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output')  # Dove scrivere i log
input_dir = os.environ.get('SM_INPUT_DIR', '/opt/ml/input/data')  # Dove si trovano i dati di input

# Elenco dei file nell'input directory per debug
print("Contenuto della directory di input:")
list_files_and_directories(input_dir)

# Specifica il nome corretto del file CSV
data_file = os.path.join(input_dir, 'vgsales.csv')  # Aggiornato con il nome corretto

# Debugging: stampa il percorso e verifica l'esistenza del file
print(f"Percorso del file di dati: {data_file}")
file_exists = os.path.isfile(data_file)
print(f"Esiste il file? {file_exists}")

if not file_exists:
    raise FileNotFoundError(f"Il file {data_file} non è stato trovato. Assicurati che sia caricato correttamente.")

# Caricamento dei dati
data = pd.read_csv(data_file, encoding='latin1')

# Pulizia dei dati
data = data.dropna(subset=['Genre', 'Year'])  # Rimuovo valori mancanti essenziali
data['Year'] = data['Year'].astype(int)  # Converto l'anno in intero

# Codifica delle variabili categoriali
label_encoder_platform = LabelEncoder()
data['Platform'] = label_encoder_platform.fit_transform(data['Platform'])

# Gestione dei valori mancanti in 'Publisher'
data['Publisher'] = data['Publisher'].fillna('Unknown')
label_encoder_publisher = LabelEncoder()
data['Publisher'] = label_encoder_publisher.fit_transform(data['Publisher'])

# Preparazione dei dati (features e target)
features = data[['Platform', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
target = LabelEncoder().fit_transform(data['Genre'])

# Divisione del dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Addestramento del modello
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Valutazione
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=LabelEncoder().fit(data['Genre']).classes_)
print(report)

# Scrittura del report nei log
report_file = os.path.join(output_dir, 'classification_report.txt')
with open(report_file, 'w') as f:
    f.write(report)

# Salvataggio del modello
model_file = os.path.join(model_dir, 'genre_classifier.pkl')
joblib.dump(model, model_file)
print(f"Modello salvato in {model_file}")
