# Import libraries
from flask import Flask, request, render_template
import joblib
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Text preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the message from the form
        message = request.form['message']
        # Preprocess the message
        processed_message = preprocess_text(message)
        # Vectorize the message
        message_vector = vectorizer.transform([processed_message])
        # Predict
        prediction = model.predict(message_vector)
        # Return the result
        result = "Spam" if prediction[0] == 1 else "Ham"
        return render_template('index.html', prediction_text=f'The message is: {result}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
