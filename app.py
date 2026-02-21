from flask import Flask , render_template , request
from tensorflow.keras.models import load_model
import numpy as np 
from PIL import Image , ImageOps

app = Flask(__name__)
model = load_model(r"D:\python_projects\DL\Neural Network For Handwritten Digits Classification\model\model.h5")

def preprocessing(image):
    new_image  = Image.open(image).convert("L")
    new_image  = new_image.resize((28 , 28))
    new_image  = np.array(new_image)
    new_image  = new_image / 255
    new_image  = new_image.reshape(1,28*28)
    return new_image

def predict(new_image):
    prediction = model.predict(new_image)
    max_pred = np.argmax(prediction)
    return max_pred

@app.route('/' , methods = ['GET','POST'])
def home():
    prediction = 0
    if request.method =='POST':
        image = request.files['image']
        pre_image = preprocessing(image)
        prediction = predict(pre_image)

    return render_template("index.html",pred_value = prediction)




if __name__ == "__main__":
    app.run(debug=True)
