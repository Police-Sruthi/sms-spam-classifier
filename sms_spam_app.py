import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and expand dataset
@st.cache_data
def load_data():
    # Load original dataset
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=["label", "message"])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Add modern SPAM messages
    extra_spam = [
        "Congratulations! You've won a free iPhone. Click here to claim: bit.ly/winfree",
        "You have been selected for a ₹10,000 Amazon voucher! Claim now.",
        "Get rich fast! Work from home and earn ₹50,000/week. No experience needed.",
        "Win money! Send WIN to 80085 now!",
        "Your Netflix account is suspended. Login here to fix: net-flix-help.com",
        "Hey! You earned 100 dollars",
        "Free recharge! Visit freemoney.biz now",
        "Urgent! Your loan is pre-approved. Call now"
    ]

    # Add modern NOT SPAM (ham) messages
    extra_ham = [
        "Are we still meeting for coffee at 5?",
        "Happy Birthday! Have a great day 🎉",
        "I'll call you once I'm done with my work.",
        "Let's catch up this weekend?",
        "Hey mom, I'm on my way home.",
        "Don't forget your appointment at 4 PM.",
        "Meeting has been moved to Monday.",
        "Yes, I submitted the assignment."
    ]

    for msg in extra_spam:
        df.loc[len(df.index)] = [1, msg]
    for msg in extra_ham:
        df.loc[len(df.index)] = [0, msg]

    return df

# Train model
@st.cache_resource
def train_model(data):
    X = data['message']
    y = data['label']
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    model = MultinomialNB()
    model.fit(X_vec, y)
    return model, vectorizer

# Predict function
def predict(model, vectorizer, message):
    vector = vectorizer.transform([message])
    result = model.predict(vector)[0]
    return "🚫 Spam" if result == 1 else "✅ Not Spam"

# Streamlit Web App
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.markdown("Check whether an SMS message is spam or not.")

# Load data and train model
data = load_data()
model, vectorizer = train_model(data)

# UI input
user_input = st.text_area("✉️ Enter your SMS message")

if st.button("Check"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        result = predict(model, vectorizer, user_input)
        if "Spam" in result:
            st.error(result)
        else:
            st.success(result)


