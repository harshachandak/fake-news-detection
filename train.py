import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("STARTING TRAINING...")

nltk.download("stopwords")
from nltk.corpus import stopwords

# =====================
# LOAD DATA
# =====================
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

# ⚡ IMPORTANT FIX: equal sampling
fake = fake.sample(n=min(len(fake), 20000), random_state=42)
true = true.sample(n=min(len(true), 20000), random_state=42)

df = pd.concat([fake, true])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[["content", "label"]]

# =====================
# CLEAN TEXT (LESS AGGRESSIVE FIX)
# =====================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["content"] = df["content"].apply(clean_text)

# =====================
# SPLIT
# =====================
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# TF-IDF (FIXED)
# =====================
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================
# MODEL (IMPORTANT FIX)
# =====================
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)

print("\n====================")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# =====================
# SAVE MODEL
# =====================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("MODEL SAVED SUCCESSFULLY ✅")