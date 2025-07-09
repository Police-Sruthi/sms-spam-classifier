import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset from URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=["label", "message"])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Train model
@st.cache_resource
def train_model(data):
    X = data['message']
    y = data['label']
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vectorized, y)
    return model, vectorizer

# Prediction
def predict(model, vectorizer, message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    return "🚫 Spam" if prediction == 1 else "✅ Not Spam"

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.markdown("Enter an SMS message below to check if it's spam or not.")

data = load_data()
model, vectorizer = train_model(data)

user_input = st.text_area("✉️ Your Message")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict(model, vectorizer, user_input)
        if "Spam" in result:
            st.error(result)
        else:
            st.success(result)
