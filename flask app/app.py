from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import io

app = Flask(__name__)
model = load_model('model.h5')  
labels = ['Biodegradable Images', 'Recyclable Images', 'Trash Images']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        # Load and preprocess image
        img = load_img(io.BytesIO(file.read()), target_size=(224, 224))
        x = img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)

        preds = model.predict(x)

        # Validate prediction shape
        if preds.size == 0 or len(preds[0]) != len(labels):
            return f"Error: Invalid model output shape {preds.shape}"

        prediction = labels[np.argmax(preds)]
        confidence = round(100 * np.max(preds), 2)

        return render_template('result.html', prediction=prediction, confidence=confidence)

    except Exception as e:
        return f" Error processing file: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=8000)
