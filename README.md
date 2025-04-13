# Next Word Prediction with LSTM - Hamlet Dataset

This project implements a Next Word Prediction model using **Long Short-Term Memory (LSTM)**, trained on the iconic **Hamlet** text by William Shakespeare. It’s my first experiment in the world of **Natural Language Processing (NLP)**, aiming to predict the next word in a sequence of text.

## 📝 **Project Overview**

In this project, I trained an LSTM model to predict the next word in a given sequence of words. The model uses the **Hamlet** text as its dataset, and the challenge is to predict the next word in a sentence based on the sequence provided.

Despite the model's **40% accuracy**, it demonstrates my initial dive into **NLP** and **Deep Learning** using **LSTM**. Due to limited resources and lack of GPU support, the model may need improvements, but this is just the beginning! 🚀

## 🔧 **Technologies Used**

- **Deep Learning**: LSTM (Long Short-Term Memory)
- **Libraries**:
  - **TensorFlow/Keras** for building and training the LSTM model.
  - **Streamlit** for creating an interactive web app for the Next Word Prediction.
  - **NumPy** for handling data.
  - **Pickle** for saving the tokenizer.
  
## 🎯 **Features**

- **LSTM-based Model**: Trained on Shakespeare's Hamlet dataset.
- **Next Word Prediction**: Predicts the next word given a sequence of words.
- **Streamlit Web App**: A simple, user-friendly interface to interact with the model.
  
## 📊 **Model Performance**

- **Accuracy**: 40% (initial run with limited resources, and without GPU support)
- **Model Type**: Sequential LSTM model with Early Stopping.

## 🛠️ **Setup Instructions**

1. Clone this repository:
   ```bash
   git clone https://github.com/priya-200/Next-Word-Prediction.git
2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```
3. Download the pre-trained LSTM model and tokenizer:
```bash
best_model.h5 (LSTM Model)

tokenizer.pickle (Tokenizer used during training)
```
4. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📈 How It Works
1. Data Preprocessing: The Hamlet text is tokenized and padded into sequences.

2. Model Training: The LSTM model is trained on the processed data to predict the next word in the sequence.

3. Prediction: Enter a sentence into the Streamlit app, and the model will predict the next word based on the provided input.

## ⚙️ Files Overview
1. best_model.h5: The trained LSTM model.

2. tokenizer.pickle: The tokenizer used to preprocess the text.

3. app.py: The Streamlit web application for next word prediction.

4. requirements.txt: Python dependencies for the project.

## 🚀 Future Enhancements
- Train the model on larger datasets for better accuracy.

- Implement GPU support to speed up the training process.

- Fine-tune the model for more accurate word predictions.

- Explore other NLP techniques such as transformers for improved results.
