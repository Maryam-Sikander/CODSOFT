# Import libraries
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Define the custom CSS style with a background image

# 1st it transforms the SMS

def transform_sms(text):
    # Step 1: Convert to lowercase and remove punctuation
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    
    # Step 2: Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Step 3: Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    
    # Step 4: Apply stemming
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    
    return ' '.join(text)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

page_bg_image = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1507963901243-ebfaecd5f2f4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1504&q=80");
background-size: cover;
}

[data-tesid="stHeader"] {
background-color: rgba(0,0,0,0);
}
</style>
"""

# Use st.markdown to apply the custom CSS
st.markdown(page_bg_image, unsafe_allow_html=True)

st.title("SMS Spam Classifier ✉ ")

input_sms =  st.text_input("Enter the Message")

if st.button('Predict'):

    transform_text = transform_sms(input_sms)
    # 2nd - Vectorize the transform text
    vectorize = tfidf.transform([transform_text])
    result = model.predict(vectorize)[0]
    # 3rd - Predict
    if result == 1:
        st.header('Spam SMS👿')
    else:
        st.header('Not a Spam SMS😊')
