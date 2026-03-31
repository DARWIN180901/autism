import os
import cv2
import sqlite3
import numpy as np
import json
import gc
from flask import Flask, request, render_template, session, redirect, url_for, g
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import tflite_runtime.interpreter as tflite  # Only use the lightweight runtime

app = Flask(__name__)
app.secret_key = 'super_secret_secure_key_123' 
app.config['UPLOAD_FOLDER'] = 'uploads'
DATABASE = 'users.db'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 1. DATABASE CONNECTION ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None: db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''CREATE TABLE IF NOT EXISTS users 
                      (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       username TEXT UNIQUE NOT NULL, 
                       password TEXT NOT NULL)''')
        db.commit()

# --- 2. LOAD TFLITE MODEL ---
TFLITE_PATH = "model/optimized_model.tflite"

try:
    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ SUCCESS: TFLite Model loaded.")
except Exception as e:
    print(f"❌ TFLite Load Error: {e}")
    interpreter = None

try:
    with open('model/class_indices.json', 'r') as f:
        class_mapping = json.load(f)
        CLASSES = {int(k): v.replace('_', ' ').title() for k, v in class_mapping.items()}
except Exception:
    CLASSES = {0: 'Autism', 1: 'Invalid', 2: 'No Autism'}

# --- 3. ML LOGIC USING TFLITE ---
def run_tflite_inference(img_array):
    """Helper to run inference without loading the full TF library"""
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def process_and_predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = run_tflite_inference(img_array)
    class_idx = np.argmax(predictions)
    return class_idx, predictions[class_idx]

def process_and_predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, max(0, total_frames - 1), 5, dtype=int)
    
    frame_predictions = []
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if current_frame in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            pred = run_tflite_inference(img_array)
            frame_predictions.append(pred)
        current_frame += 1
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
        if interpreter is None: return "Model not loaded."

        if ext in ['mp4', 'avi', 'mov', 'webm']:
            class_idx, conf = process_and_predict_video(filepath)
        elif ext in ['jpg', 'jpeg', 'png']:
            class_idx, conf = process_and_predict_image(filepath)
        else:
            return "Unsupported file type."

        label = CLASSES.get(class_idx, "Unknown")
        return render_template('result.html', label=label, confidence=float(conf) * 100)
    except Exception as e:
        return f"Analysis error: {str(e)}"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
        gc.collect()

init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)