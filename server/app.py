from flask import Flask, request, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import datetime
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
app.secret_key = 'process.env.JWT_SECRET'
mongo_uri = 'mongodb+srv://krishsoni:2203031050659@paytm.aujjoys.mongodb.net/'

client = MongoClient(mongo_uri)
db = client['Drive']
users_collection = db['users']
uploads_collection = db['uploads']

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    return jsonify({"message": "Login successful"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    username = request.form.get('username')
    password = request.form.get('password')
    file = request.files['file']

    if not username or not password or not file:
        return jsonify({"error": "Missing required parameters"}), 400

    user = users_collection.find_one({"username": username})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid username or password"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        upload_record = {
            "username": username,
            "filename": filename,
            "filepath": filepath,
            "upload_date": datetime.datetime.utcnow()
        }
        uploads_collection.insert_one(upload_record)

        return jsonify({"message": "File uploaded successfully"}), 201

    return jsonify({"error": "File type not allowed"}), 400

@app.route('/dashboard', methods=['GET'])
def dashboard():
    username = request.args.get('username')
    password = request.args.get('password')

    if not username or not password:
        return jsonify({"error": "Missing required parameters"}), 400

    user = users_collection.find_one({"username": username})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid username or password"}), 400

    uploads = uploads_collection.find({"username": username})
    upload_list = [{"filename": upload["filename"], "upload_date": upload["upload_date"]} for upload in uploads]

    return jsonify({"message": f"Welcome to your dashboard, {username}!", "uploads": upload_list})

if __name__ == '__main__':
    app.run(debug=True)
