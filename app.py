from flask import Flask , render_template , request


app = Flask(__name__)


def pred(img):
    from tensorflow.keras.models import load_model
    model = load_model(r"D:\python_projects\DL\Neural Network For Handwritten Digits Classification\model\model.h5")
    pred_value = model.predict(img)
    return pred_value

@app.route('/' , methods = ['GET','POST'])
def home():
    prediction = 0
    if request.method =='POST':
        image = request.files['image']
        prediction = pred(image)

    return render_template("index.html",pred_value = prediction)




if __name__ == "__main__":
    app.run(debug=True)
