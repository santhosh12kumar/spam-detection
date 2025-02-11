# Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string

# Load and Preprocessed Data
# Load Dataset
df = pd.read_csv('spam.csv', encoding='latin-1') # Put your dataset path in place of this
df = df[['v1', 'v2']] # Take the needed columns, label and text
df.columns = ['label', 'message']

# Convert labels into Binary spam=1, ham=0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Text pre-processing
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocessing(text):
    # Remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    return text

df['message'] = df['message'].apply(preprocessing)

# Feature Extraction
# Transform Text into Not From Count Vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Building the Model
# Fit model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Test the Model
# Predict against the test data
y_pred = model.predict(X_test)

# Validate 
accuracy= accuracy_score(y_test,y_pred)
print("Accuracy: ", accuracy * 100, "%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
