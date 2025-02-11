# spam-detection

## Overview
This project is a **Spam Detection Web Application** built using **Python, Flask, and Scikit-Learn**. It classifies text messages as either **Spam** or **Ham (Not Spam)** using a **Naïve Bayes Classifier**. The model is trained on the `spam.csv` dataset and then deployed as a web app using Flask.

## Features
- **Preprocesses text messages** (removes punctuation and stopwords).
- **Uses CountVectorizer** to convert text into numerical data.
- **Trains a Naïve Bayes classifier** for spam detection.
- **Deploys the model using Flask** to predict if a message is spam or ham.

## Installation & Setup
### **Prerequisites**
Ensure you have **Python 3.7+** installed. You also need to install the required Python libraries.

### **1. Clone the Repository**
```sh
https://github.com/your-repository/spam-detection-flask.git
cd spam-detection-flask
```

### **2. Install Dependencies**
Run the following command to install the required packages:
```sh
pip install -r requirements.txt
```

### **3. Download NLTK Stopwords**
Run this in Python to download the stopwords dataset:
```python
import nltk
nltk.download('stopwords')
```

### **4. Train and Save the Model**
Run the `train_model.py` script (if available) to train the model and save it.
```sh
python train_model.py
```
This will generate `spam_model.pkl` and `vectorizer.pkl`.

### **5. Run the Flask App**
To start the web application, execute:
```sh
python app.py
```
Then, open your browser and go to:
```
http://127.0.0.1:5000/
```

## Project Structure
```
├── app.py               # Flask application file
├── spam.csv             # Dataset for training
├── spam_model.pkl       # Trained Naïve Bayes model
├── vectorizer.pkl       # Fitted CountVectorizer
├── templates/
│   ├── index.html       # Web interface
├── static/
│   ├── styles.css       # CSS (if applicable)
├── requirements.txt     # Required dependencies
├── README.md            # Project documentation
```

## How It Works
1. **Load Dataset**: Reads and preprocesses the dataset (`spam.csv`).
2. **Text Preprocessing**: Removes punctuation and stopwords.
3. **Feature Extraction**: Uses `CountVectorizer` to convert text into numeric vectors.
4. **Train Model**: Uses `MultinomialNB` (Naïve Bayes) classifier.
5. **Save Model**: Stores trained model and vectorizer as `.pkl` files.
6. **Deploy with Flask**: Accepts user input, preprocesses it, makes predictions, and displays results.

## Usage
1. Enter a message in the web app.
2. Click "Predict" to check if the message is Spam or Ham.
3. The result is displayed on the page.

## Example Prediction
**Input:** "Congratulations! You have won a free lottery. Claim now."
**Output:** "Spam"

**Input:** "Hey, are we meeting tomorrow?"
**Output:** "Ham"

## Technologies Used
- **Python 3**
- **Flask** (for web app)
- **NLTK** (for text processing)
- **Scikit-Learn** (for machine learning)
- **Pandas & NumPy** (for data handling)

## Future Improvements
- Integrate deep learning models for better accuracy.
- Add a database to store messages.
- Deploy on a cloud platform like **Heroku** or **AWS**.

## License
This project is open-source and available under the **MIT License**.

---

### Author
Developed by @Santhoshkumar SK

For any queries, contact: sksanthosh@gmail.com

