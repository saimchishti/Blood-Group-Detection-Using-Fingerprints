import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os

# Initialize Flask app
app = Flask(__name__)

# Define a folder for storing uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
def load_model(model_path):
    """
    Load the pre-trained model from the given file path.
    """
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image_path, target_size):
    """
    Preprocess the input image to match the model's expected input format.
    - image_path: Path to the fingerprint image
    - target_size: The size the model expects for the input image (e.g., (64, 64))
    """
    # Load the image
    img = Image.open(image_path).convert('RGB')  # Ensure the image has 3 channels (RGB)
    # Resize the image
    img = img.resize(target_size)
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_blood_group(image_path, model_path, target_size=(64, 64)):
    """
    Predict the blood group from a fingerprint image using the trained model.
    - image_path: Path to the fingerprint image
    - model_path: Path to the saved model
    - target_size: Expected input size of the model

    Returns:
        Predicted blood group label
    """
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    processed_image = preprocess_image(image_path, target_size)

    # Predict using the model
    prediction = model.predict(processed_image)

    # Decode the prediction (assuming model predicts probabilities for classes)
    classes = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']  # Example classes
    predicted_class = classes[np.argmax(prediction)]

    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Predict the blood group from the image
        try:
            predicted_blood_group = predict_blood_group(file_path, model_path="100EPOCHhigh_accuracy_blood_group_model.h5")
            return jsonify({'result': predicted_blood_group})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Create the uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)
