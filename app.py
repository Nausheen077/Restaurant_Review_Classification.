import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (only first time)
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("restaurant_model.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

ps = PorterStemmer()

# Page title
st.title("🍽️ Restaurant Review Sentiment Analysis")

st.write("Enter a restaurant review below and click Predict.")

# User input
review = st.text_area("Enter your review:")

# Prediction button
if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess the review
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()

        review = [
            ps.stem(word)
            for word in review
            if word not in set(stopwords.words('english'))
        ]

        review = ' '.join(review)

        review = cv.transform([review]).toarray()

        prediction = model.predict(review)

        if prediction[0] == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")