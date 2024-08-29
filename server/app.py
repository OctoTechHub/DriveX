from flask import Flask, request, jsonify, session, send_from_directory
import numpy as np
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import datetime
from flask_cors import CORS
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import io
import json
app = Flask(__name__)
CORS(app, origins=['http://localhost:5173','http://localhost:5174'], methods=['GET', 'POST', 'PUT', 'DELETE'], supports_credentials=True)
app.secret_key = 'process.env.JWT_SECRET'
mongo_uri = 'mongodb+srv://krishsoni:2203031050659@paytm.aujjoys.mongodb.net/'

client = MongoClient(mongo_uri)
db = client['Drive']
users_collection = db['users']
uploads_collection = db['uploads']

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docs', 'docx', 'json', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Initialize model parameters
model_path = 'server/models/simple_model.pth'
input_dim = 1000  # Replace with the actual input dimension
output_dim = 10   # Replace with the actual number of classes
model = SimpleModel(input_dim, output_dim)

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Load vectorizer and label_encoder from files if they exist
vectorizer_path = 'vectorizer.pkl'
label_encoder_path = 'label_encoder.pkl'

def load_vectorizer():
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError("Vectorizer file not found.")

def load_label_encoder():
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError("Label encoder file not found.")

def fit_and_save_vectorizer_and_encoder():
    # Sample training data and labels
    texts = ["Sample text for training", "Another sample text", "More text data"]
    labels = ["category1", "category2", "category1"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Save vectorizer and label encoder
    with open(vectorizer_path, 'wb') as file:
        pickle.dump(vectorizer, file)
    
    with open(label_encoder_path, 'wb') as file:
        pickle.dump(label_encoder, file)

    return vectorizer, label_encoder

# Load vectorizer and label_encoder
try:
    vectorizer = load_vectorizer()
    label_encoder = load_label_encoder()
except FileNotFoundError:
    # Fit and save if not found
    vectorizer, label_encoder = fit_and_save_vectorizer_and_encoder()

@app.route('/',methods=['GET'])
def home():
    return "Welcome to the Drive API!"


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    phone = data.get('phone')
    email = data.get('email')

    if not username or not password or not (phone or email):
        return jsonify({"error": "Missing required parameters"}), 400

    existing_user = users_collection.find_one({"username": username})
    if existing_user:
        return jsonify({"error": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    user = {
        "username": username,
        "password": hashed_password,
        "phone": phone,
        "email": email
    }
    users_collection.insert_one(user)

    return jsonify({"message": "User created successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Missing required parameters"}), 400

    user = users_collection.find_one({"username": username})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid username or password"}), 400

    session['username'] = username
    session['password'] = password

    return jsonify({"message": "Login successful"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file part"}), 400

    if not session.get('username'):
        return jsonify({"error": "User not logged in"}), 401

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"error": "Internal server error"}), 500

    upload_record = {
        "username": session.get('username'),
        "filename": filename,
        "filepath": filepath,
        "upload_date": datetime.datetime.utcnow()
    }
    uploads_collection.insert_one(upload_record)

    return jsonify({"message": "File uploaded successfully"}), 201

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if 'username' not in session or 'password' not in session:
        return jsonify({"error": "User not logged in"}), 401

    username = session['username']
    password = session['password']

    user = users_collection.find_one({"username": username})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid username or password"}), 401

    uploads = list(uploads_collection.find({"username": username}))
    upload_list = [{"filename": upload["filename"], "upload_date": upload["upload_date"]} for upload in uploads]

    return jsonify({"message": f"Welcome to your dashboard, {username}!", "uploads": upload_list})

model = None
vectorizer = None
label_encoder = None

def load_model_and_vectorizer():
    global model, vectorizer, label_encoder
    # Load the model
    model = SimpleModel(input_dim, output_dim)  # Initialize model
    model.load_state_dict(torch.load('simple_model.pth'))  # Load model state
    model.eval()  # Set model to evaluation mode

    # Load vectorizer and label encoder
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

@app.route('/classify', methods=['POST'])
def classify_file():
    try:
        if model is None or vectorizer is None or label_encoder is None:
            load_model_and_vectorizer()  # Ensure model and vectorizer are loaded

        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        print(f"Received text for classification: {text}")

        # Transform the new text into a vector using the loaded vectorizer
        new_text_vector = torch.tensor(vectorizer.transform([text]).toarray(), dtype=torch.float32)

        # Make a prediction using the loaded model
        prediction = model(new_text_vector)
        _, predicted_class = torch.max(prediction, dim=1)
        category = label_encoder.inverse_transform(predicted_class.numpy())

        return jsonify({'category': category[0]}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

categories = {
    'pdf': ['document', 'pdf'],
    'image': ['image', 'picture', 'jpg', 'jpeg', 'png', 'gif'],
    'word': ['word', 'doc', 'docx'],
    'excel': ['excel', 'xls', 'xlsx'],
    'powerpoint': ['powerpoint', 'ppt', 'pptx']
}

def classify_text(text):
    text = text.lower()
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'unknown'

@app.route('/simple_classify', methods=['POST'])
def simple_classify():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        print(f"Received text for classification: {text}")

        # Perform manual classification
        category = classify_text(text)

        return jsonify({'category': category}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
        
    
if __name__ == '__main__':
    app.run(debug=True)
