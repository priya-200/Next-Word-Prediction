import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Load the LSTM Model (only once)
if 'model' not in st.session_state:
    st.session_state.model = load_model('best_model.h5')

# Load the Tokenizer (only once)
if 'tokenizer' not in st.session_state:
    with open('tokenizer.pickle', 'rb') as handle:
        st.session_state.tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre', truncating='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")

if st.button("Predict Next Word"):
    # Ensure the model is loaded only once per session
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
