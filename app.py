import streamlit as st
import string
import pickle
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()   # step 1
    text = nltk.word_tokenize(text, preserve_line=True) # step 2

    y = []
    for i in text:   # step 3 
        if i.isalnum():
            y.append(i)
            
    text=y[:] # List is mutable data type, we should cloning it like this instead of assigning it
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation: # step 4
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:   
        y.append(ps.stem(i))   # step 5
    
    return " ".join(y)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)

    # 2. vectorize
    vectorized_sms = tfidf.transform([transformed_sms])

    # 3. prediction
    result = model.predict(vectorized_sms)[0]

    # 4. Display the result
    st.write("Result:", "Spam" if result == 1 else "Not Spam")