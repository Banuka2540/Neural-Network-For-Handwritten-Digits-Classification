from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = load_model('model/mnist_model.h5')  # Load Keras model once

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_digit(filepath):
    img = Image.open(filepath).convert('L')
    img = img.resize((28,28))
    img_array = np.array(img).reshape(1,28,28,1)
    img_array = img_array / 255.0
    pred = model.predict(img_array)
    return np.argmax(pred, axis=1)[0]

@app.route('/', methods=['GET','POST'])
def home():
    prediction = None
    if request.method == 'POST':
        if 'digit_image' not in request.files:
            return "No file uploaded"
        file = request.files['digit_image']
        if file.filename == '':
            return "No file selected"
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_digit(filepath)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)