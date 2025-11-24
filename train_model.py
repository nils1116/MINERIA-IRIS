import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. Cargar dataset
df = pd.read_csv("Iris.csv")

# 2. Preprocesamiento
df = df.drop(columns=["Id"], errors="ignore")

X = df.drop("Species", axis=1)
y = df["Species"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 3. Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar modelo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Evaluar
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# 6. Guardar modelo y encoder
joblib.dump(model, "iris_model.pkl")
joblib.dump(encoder, "encoder.pkl")
