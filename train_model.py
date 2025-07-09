import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['message']
y = df['label']

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved.")
