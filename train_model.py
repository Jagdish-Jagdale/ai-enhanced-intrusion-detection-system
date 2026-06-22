import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

print("Loading dataset...")
data = pd.read_csv("web_attacks_balanced.csv")

target_column = "Label"
selected_features = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets'
]

X = data[selected_features]
y = data[target_column]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Training RandomForestClassifier...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

print("Saving model...")
joblib.dump(model, 'random_forest_model_4_features.joblib')
print("Model saved as 'random_forest_model_4_features.joblib'")

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
