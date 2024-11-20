import streamlit as st
import joblib

# Load the trained models and vectorizer
nb_model = joblib.load('nb_model.pkl')             # Path to your Naive Bayes model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Path to your vectorizer

# Streamlit App
st.title("Spam Email Detection")
st.write("Enter an email below to classify it as Spam or Ham.")

# Input text area
email_input = st.text_area("Email Content", height=200)

if st.button("Classify"):
    if email_input.strip() == "":
        st.warning("Please enter some email content!")
    else:
        # Preprocess input
        email_vectorized = tfidf_vectorizer.transform([email_input])
        prediction = nb_model.predict(email_vectorized)
        
        # Map prediction to label
        label_map = {'ham': "Ham", 'spam': "Spam"}
        st.subheader(f"Prediction: {label_map[prediction[0]]}")
