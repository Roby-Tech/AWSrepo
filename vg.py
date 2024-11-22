import os
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# Funzione per scaricare un file da S3
def download_file_from_s3(bucket_name, file_key, local_path):
    print(f"Scaricamento del file S3 {file_key} dal bucket {bucket_name}...")
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, file_key, local_path)
    print(f"File scaricato e salvato in: {local_path}")

# Variabili d'ambiente
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')  # Dove salvare il modello
output_dir = os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output')  # Dove scrivere i log
input_dir = os.environ.get('SM_INPUT_DIR', '/opt/ml/input/data')  # Dove si trovano i dati di input
s3_bucket = os.environ.get('S3_BUCKET_NAME', None)  # Nome del bucket S3
s3_key = os.environ.get('S3_FILE_KEY', None)  # Percorso del file S3

# Verifica delle variabili d'ambiente
if not s3_bucket or not s3_key:
    raise ValueError("Le variabili S3_BUCKET_NAME e S3_FILE_KEY devono essere definite.")

# Percorso locale per salvare il file scaricato da S3
local_file_path = os.path.join(input_dir, 'vgsales.csv')

# Scaricamento del file da S3
download_file_from_s3(s3_bucket, s3_key, local_file_path)

# Verifica del file scaricato
if not os.path.isfile(local_file_path):
    raise FileNotFoundError(f"Il file {local_file_path} non è stato trovato dopo il download da S3.")

print(f"Percorso del file scaricato: {local_file_path}")

# Caricamento dei dati
data = pd.read_csv(local_file_path, encoding='latin1')

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

# Normalizzazione delle features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Conversione del target in formato categorico (one-hot encoding)
target = to_categorical(target)

# Divisione del dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Costruzione del modello Keras
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')  # Numero di classi = numero di colonne in y_train
])

# Compilazione del modello
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Addestramento del modello
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Salvataggio del modello nel formato .h5
model_file = os.path.join(model_dir, 'genre_classifier.keras')
model.save(model_file)
print(f"Modello salvato in {model_file}")

# Generazione di un report nei log
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true_classes = y_test.argmax(axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=LabelEncoder().fit(data['Genre']).classes_)
print(report)

# Scrittura del report nei log
report_file = os.path.join(output_dir, 'classification_report.txt')
with open(report_file, 'w') as f:
    f.write(report)
