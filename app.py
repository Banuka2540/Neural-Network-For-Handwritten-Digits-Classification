from flask import Flask , render_template , request
import pickle 
import numpy as np
from PIL import Image

app = Flask(__name__)


filename = 'Model/model.pickle'
with open(filename,'rb') as file :
    model = pickle.load(file)

def predict(image):
    pred_image = model.predict(image)   
    return pred_image

def preprocessing(image):
        new_image  = Image.open(image).convert("L")
        new_image  = new_image.resize((28 , 28))
        new_image  = np.array(new_image)
        new_image  = new_image / 255
        new_image  = new_image.reshape(1,28*28)
        return new_image


@app.route("/",methods=['GET','POST'])
def Home():
    pred = 0 
    if request.method == 'POST':
        if 'input-image' not in request.files:
            return render_template("index.html", pred_number="No file uploaded") 
        else :
            image = request.files['input-image']
        image = preprocessing(image)
        pred = predict(image)
        pred = int(pred[0])

    return render_template("index.html",pred_number = pred )

if __name__ == '__main__' :
    app.run()