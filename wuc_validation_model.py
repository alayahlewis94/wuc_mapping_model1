import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

DATA_FILE = "synthetic_23R_no_labels_copy.csv"

# Load CSV
df = pd.read_csv(DATA_FILE)

# Clean
df["combined_text"] = df["combined_text"].fillna("").astype(str)
df["entered_wuc"] = df["entered_wuc"].astype(str)

X = df["combined_text"]
y = df["entered_wuc"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=8000)),
    ("clf", LogisticRegression(max_iter=2000))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)
confidence = probs.max(axis=1)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

# Build review queue
results = pd.DataFrame({
    "combined_text": X_test.values,
    "entered_wuc": y_test.values,
    "predicted_wuc": y_pred,
    "confidence": confidence
})

THRESHOLD = 0.75
results["flag_for_review"] = (
    (results["entered_wuc"] != results["predicted_wuc"]) &
    (results["confidence"] >= THRESHOLD)
)

results.to_csv("validation_review_queue.csv", index=False)

print("\nSaved validation_review_queue.csv")