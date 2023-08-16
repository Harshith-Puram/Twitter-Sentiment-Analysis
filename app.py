import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
import torch

app = Flask(__name__)
model = pickle.load(open('C:/Users/harsh/OneDrive/Documents/NITK_Work/model.pkl', 'rb'))

# Load NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Map sentiment labels to numerical values
sentiment_mapping = {'negative': 0, 'neutral': 2, 'positive': 1}

def preprocess_text(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

@app.route('/')
def home():
    return render_template('index.html')

from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tweet = request.form['tweet']  # Assuming the input field is named 'tweet'
    
    encoded_input = tokenizer(tweet, return_tensors='pt', padding=True, truncation=True)
    # Move input tensors to GPU
    for key in encoded_input:
        encoded_input[key] = encoded_input[key].to('cuda')
    
    with torch.no_grad():
        output = model(**encoded_input)
        
    logits = output.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    sentiment_label = {0: 'negative', 1: 'positive', 2: 'neutral'}

    prediction_text = f"The sentiment of the tweet '{tweet}' is '{sentiment_label[predicted_label]}'."

    return render_template('index.html', prediction_text=prediction_text)


'''@app.route('/predict', methods=['POST'])
def predict():
    
    #For rendering results on HTML GUI
    
    tweet = request.form['tweet']  # Assuming the input field is named 'tweet'
    
    encoded_input = tokenizer(tweet, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**encoded_input)
        
    logits = output.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    sentiment_label = {0: 'negative', 1: 'positive', 2: 'neutral'}

    prediction_text = f"The sentiment of the tweet '{tweet}' is '{sentiment_label[predicted_label]}'."

    return render_template('index.html', prediction_text=prediction_text)


@app.route('/predict', methods=['POST'])
def predict():
    
    #For rendering results on HTML GUI
    
    tweet = request.form['tweet']
    processed_tweet = preprocess_text(tweet)
    prediction = model.predict([processed_tweet])

    sentiment_label = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(prediction[0])]

    return render_template('index.html', prediction_text=f'Tweet Sentiment: {sentiment_label}')'''

@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    processed_tweet = preprocess_text(data['tweet'])
    prediction = model.predict([processed_tweet])

    sentiment_label = list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(prediction[0])]

    return jsonify({'sentiment': sentiment_label})

if __name__ == "__main__":
    app.run(debug=True)
