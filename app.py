from flask import Flask , render_template , request
import pickle 
app = Flask(__name__)

def pred(image):
    filename = 'model/model.pickle'
    with open(filename,'rb') as file :
        model = pickle.load(file)
    pred_value = model.predict(image)
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
