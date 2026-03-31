import os
import cv2
import sqlite3
import numpy as np
import json
from flask import Flask, request, render_template, session, redirect, url_for, g
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
app.secret_key = 'super_secret_secure_key_123' 
app.config['UPLOAD_FOLDER'] = 'uploads'
DATABASE = 'users.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 1. DATABASE CONNECTION LOGIC ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       username TEXT UNIQUE NOT NULL, 
                       password TEXT NOT NULL)''')
        db.commit()

# --- 2. LOAD MODELS ---
IMAGE_MODEL_PATH = 'model/optimized_model.h5'
VIDEO_MODEL_PATH = 'model/video_optimized_model.h5'

image_model = None
video_model = None

try:
    if os.path.exists(IMAGE_MODEL_PATH):
        image_model = load_model(IMAGE_MODEL_PATH)
        print("✅ Image model loaded.")
except Exception as e:
    print(f"❌ Error loading image model: {e}")

try:
    if os.path.exists(VIDEO_MODEL_PATH):
        video_model = load_model(VIDEO_MODEL_PATH)
        print("✅ Video model loaded.")
except Exception as e:
    print(f"❌ Error loading video model: {e}")

try:
    with open('model/class_indices.json', 'r') as f:
        class_mapping = json.load(f)
        CLASSES = {int(k): v.replace('_', ' ').title() for k, v in class_mapping.items()}
except FileNotFoundError:
    CLASSES = {0: 'Autism', 1: 'Invalid', 2: 'No Autism'} 

# --- 3. MACHINE LEARNING LOGIC ---
def process_and_predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = image_model.predict(image_array)[0]
    class_idx = np.argmax(predictions)
    return class_idx, predictions[class_idx]

def process_and_predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % 30 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = video_model.predict(img_array, verbose=0)[0]
            frame_predictions.append(pred)
        frame_count += 1
    cap.release()
    if frame_predictions:
        avg_predictions = np.mean(frame_predictions, axis=0)
        class_idx = np.argmax(avg_predictions)
        return class_idx, avg_predictions[class_idx]
    return 0, 0.0 

# --- 4. ROUTES ---
@app.route('/')
def index():
    return redirect(url_for('dashboard')) if 'user_id' in session else redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username, password = request.form['username'], request.form['password']
        db = get_db()
        try:
            db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, generate_password_hash(password)))
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Username already exists."
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username, password = request.form['username'], request.form['password']
        user = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session.update({'user_id': user['id'], 'username': user['username']})
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/predict_page')
def predict_page():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session: return redirect(url_for('login'))
    if 'file' not in request.files: return "No file part"
    
    file = request.files['file']
    if file.filename == '': return "No selected file"
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    ext = filename.rsplit('.', 1)[1].lower()

    try:
        if ext in ['mp4', 'avi', 'mov', 'webm']:
            if video_model is None: return "Video model not loaded."
            class_idx, conf = process_and_predict_video(filepath)
        elif ext in ['jpg', 'jpeg', 'png']:
            if image_model is None: return "Image model not loaded."
            class_idx, conf = process_and_predict_image(filepath)
        else:
            return "Unsupported file type."

        label = CLASSES.get(class_idx, "Unknown")
        return render_template('result.html', label=label, confidence=conf * 100)
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)