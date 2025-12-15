# webapp/app.py
from flask import Flask, request, render_template, jsonify
import os
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from src.dataset import preprocess_image

UPLOAD_FOLDER = 'webapp/uploads'
ALLOWED_EXT = {'png','jpg','jpeg'}

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.environ.get('MODEL_PATH', 'outputs/model-final')  # set when running
CLASSES_JSON = os.environ.get('CLASSES_JSON', 'outputs/classes.json')
IMG_SIZE = (224,224)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    import json
    with open(CLASSES_JSON,'r') as f:
        classes = json.load(f)
    return model, classes

model, classes = load_model_and_classes()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'no selected file'}), 400
    if f and allowed_file(f.filename):
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        image = preprocess_image(path, IMG_SIZE)
        image = tf.expand_dims(image, axis=0)
        preds = model.predict(image)[0]
        idx = int(np.argmax(preds))
        return jsonify({
            'label': classes[idx],
            'confidence': float(preds[idx]),
            'all_probs': preds.tolist()
        })
    else:
        return jsonify({'error': 'file not allowed'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
