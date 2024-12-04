import pickle
import tensorflow as tf
import re
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('viral_model.keras')

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Flask app setup
app = Flask(__name__)

# Function to preprocess the title
def preprocess_title(title):
    # Add any preprocessing steps you need here
    title = title.lower()
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)  # Remove non-alphanumeric characters
    title = title.strip()
    return title

# Function to predict viral probability
def predict_viral_probability(title):
    cleaned_title = preprocess_title(title)
    title_vectorized = vectorizer.transform([cleaned_title]).toarray()
    probability = model.predict(title_vectorized)[0][0]
    return probability

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and predict viral probability
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        viral_probability = predict_viral_probability(title)
        return render_template('index.html', title=title, viral_probability=viral_probability)

if __name__ == "__main__":
    app.run(debug=True)
