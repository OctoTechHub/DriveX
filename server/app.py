from flask import Flask, request, jsonify, session, send_from_directory
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import datetime
from flask_cors import CORS
from fastai.text.all import load_learner
import torch

app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'], methods=['GET', 'POST', 'PUT', 'DELETE'], supports_credentials=True)
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

class CustomCrossEntropyLoss(torch.nn.Module):
    def forward(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)

def custom_accuracy(preds, targets):
    return (preds.argmax(dim=-1) == targets).float().mean()

# Path to your model
model_path = 'D:/Python/Projects/project-mlsa/server/models/file_classifier.pkl'
learn = None
try:
    learn = load_learner(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

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

@app.route('/classify', methods=['POST'])
def classify_file():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        print(f"Received text for classification: {text}")

        if not learn:
            raise RuntimeError('Model not loaded')

        print("Predicting...")
        prediction = learn.predict(text)
        print(f"Prediction: {prediction}")

        return jsonify({'category': prediction[0]}), 200

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
