import streamlit as st
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Function to load dataset
def load_dataset(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as z:
        file_name = z.namelist()[0]  # Get the first file inside ZIP
        with z.open(file_name) as f:
            df = pd.read_csv(f, encoding='latin-1')
    return df

# Function to train the model
def train_model(df):
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    label_mapping = {'spam': 1, 'ham': 0}
    df = df[df['label'].isin(label_mapping)]
    df['label'] = df['label'].map(label_mapping)
    df = df.dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    
    y_pred = nb.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return vectorizer, nb, accuracy, report

# Function to classify an email
def classify_email(email, vectorizer, model):
    input_tfidf = vectorizer.transform([email])
    prediction = model.predict(input_tfidf)
    return "Spam" if prediction[0] == 1 else "Ham"

# Streamlit UI
st.title("Spam Detector App")

uploaded_file = st.file_uploader("Upload a ZIP file containing the dataset", type="zip")

if uploaded_file:
    st.write("Processing dataset...")
    df = load_dataset(uploaded_file)
    vectorizer, model, accuracy, report = train_model(df)
    
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    st.text(f"Classification Report:\n{report}")

    email_text = st.text_area("Enter email text for classification:")
    
    if st.button("Classify Email"):
        if email_text:
            prediction = classify_email(email_text, vectorizer, model)
            st.success(f"Prediction: {prediction}")
        else:
            st.warning("Please enter an email text for classification.")
