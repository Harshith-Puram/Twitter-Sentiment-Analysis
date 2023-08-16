# Twitter Sentiment Analysis

Welcome to the Twitter Sentiment Analysis repository! This project is dedicated to harnessing the power of natural language processing and machine learning to analyze the sentiment of tweets. By leveraging advanced techniques, we aim to classify tweets as positive, negative, or neutral based on their underlying sentiment.

## User Interface
![HOME PAGE](https://github.com/Harshith-Puram/Twitter-Sentiment-Analysis/blob/main/sentiment_analysis.png)

## Project Overview

In a world where social media platforms are teeming with opinions and emotions, understanding the sentiment behind tweets can provide valuable insights. Our Twitter Sentiment Analysis project utilizes cutting-edge techniques to decipher the emotional tone of tweets, helping individuals and businesses gain a deeper understanding of public sentiment.

## Project Components

Our repository is structured to cover every aspect of the Twitter Sentiment Analysis project:

- *Dataset:* Explore the `dataset` folder to find the tweet data used for training and testing our sentiment analysis model.
- *Model Development:* The `model_train.ipynb` Jupyter Notebook details the steps we took to preprocess data, build the sentiment analysis model, and evaluate its performance.
- *Web Application:* Check out the `app.py` file, which is the core of our Flask web application for deploying the trained model. Users can input tweets and receive sentiment predictions in real-time.
- *Frontend:* The `templates` folder holds the HTML template for the web interface, while the `static/css` folder contains the styles for the web application.
- *Documentation:* The `README.md`file provides comprehensive information about the project, its usage, installation instructions, and more.

We hope our project sparks your curiosity and helps you delve into the fascinating world of sentiment analysis on Twitter!

## About the Model

In our pursuit of accurate sentiment analysis, we harnessed the capabilities of BERT (Bidirectional Encoder Representations from Transformers), a cutting-edge transformer-based model developed by Google. BERT has revolutionized natural language processing tasks, and we integrated it into our sentiment analysis pipeline to achieve even higher accuracy and contextual understanding.

## Model Accuracy

Our commitment to accuracy led us to integrate BERT (Bidirectional Encoder Representations from Transformers) into our sentiment analysis model. Leveraging the power of BERT's contextual understanding and semantic relationships, we achieved a significant advancement in sentiment prediction accuracy.

*BERT-Powered Accuracy: 84.91%*

After rigorous training, fine-tuning, and evaluation, our sentiment analysis model powered by BERT demonstrated an impressive accuracy rate of 84.91%. This remarkable performance showcases the impact of leveraging state-of-the-art techniques for sentiment analysis.

## Technology Stack

Our Twitter Sentiment Analysis project is built upon a robust technology stack that encompasses both the backend and frontend components. This stack ensures efficient development, seamless deployment, and an engaging user experience.

### Backend

*Framework:* Flask


*Python Version:* 3.9.6


*Dependencies:*

- *numpy==1.21.2*
- *pandas==1.3.3*
- *scikit-learn==0.24.2*
- *torch*
- *transformers* 
- *matplotlib*

### Frontend

- *HTML* 

- *CSS*
  
## How to Use

Interacting with our Twitter Sentiment Analysis project is straightforward. Follow these steps to harness the power of sentiment analysis on Twitter data:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the necessary dependencies using the provided `requirements.txt` file.
4. Run the Flask application using the command:
   
```
 python app.py
```

5. Access the model's user interface by opening a web browser and navigating to `http://localhost:5000`.

On the user interface, provide input values for the following parameters:
- Enter your tweet 
- Click on predict

After submitting the input, the model will perform its predictions and display the sentiment.

## Requirements

Ensure you have the required dependencies installed by running:
```
pip install -r requirements.txt
```

## Contributing

Contributions to enhance and extend the capabilities of the Crop Prediction Model are highly welcome. If you wish to contribute, please adhere to these steps:

1. Fork the repository.
2. Create a new branch dedicated to your feature or improvement.
3. Implement your changes and ensure comprehensive testing.
4. Submit a pull request, detailing the modifications and the underlying rationale.

## Credits

This machine-learning marvel was brought to life by the collaborative efforts of [Harshith Puram](https://github.com/Harshith-Puram) and [Druvika Nuthalapati](https://github.com/druvikan), fueled by the demand for accurate and data-driven sentiment analysis of social media content.

## Contact

For inquiries, issues, or suggestions, please feel free to reach out via harshithppuram@gmail.com or druvikan@gmail.com.
