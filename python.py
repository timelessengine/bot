from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import sqlite3
import json
import os

# Load machine learning model
model = load_model('model.h5')

# Load tokenizer
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Define max length of input sequence
max_input_length = 50

# Create Flask app
app = Flask(__name__)

# Set up SQLite database
db_file = 'data.db'
if not os.path.exists(db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE messages (input_text text, output_text text)''')
    conn.commit()
    conn.close()

# Define prediction function
def predict(input_text):
    # Tokenize input text
    input_tokens = tokenizer.texts_to_sequences([input_text])
    # Pad input sequence
    input_sequence = pad_sequences(input_tokens, maxlen=max_input_length, padding='post')
    # Make prediction
    output = model.predict(input_sequence)
    # Convert prediction to text
    output_text = tokenizer.sequences_to_texts([np.argmax(output)])[0]
    return output_text

# Define endpoint for receiving input text
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input text from form submission
        input_text = request.form['input_text']
        # Make prediction
        output_text = predict(input_text)
        # Save input and output to database
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute('''INSERT INTO messages (input_text, output_text) VALUES (?, ?)''', (input_text, output_text))
        conn.commit()
        conn.close()
        # Render output template with predicted output
        return render_template('index.html', output_text=output_text)
    else:
        # Render input form template
        return render_template('input.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
