import pandas as pd
import numpy as np
import re, string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Sample mini dataset (for demo only)
data = {
    "text": [
        "Breaking News: Government launches new scheme for farmers",
        "Shocking! Celebrity caught in fake scandal",
        "NASA confirms water on Mars",
        "Doctors warn about fake COVID cures spreading online",
        "Elections are rigged, says random social media post",
        "Local school wins national science competition",
    ],
    "label": ["REAL", "FAKE", "REAL", "FAKE", "FAKE", "REAL"]
}

df = pd.DataFrame(data)

# 2. Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["clean_text"] = df["text"].apply(clean_text)

# 3. Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# 4. Train model
model = LogisticRegression()
model.fit(X, y)

# 5. Save model + vectorizer
joblib.dump(model, "src/fake_news_model.pkl")
joblib.dump(vectorizer, "src/vectorizer.pkl")

print("✅ Model and Vectorizer saved inside src/")