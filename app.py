from flask import Flask , render_template , request
import pickle

app = Flask(__name__)

def predict(number):
    filename = 'Model/model.pickle'
    with open(filename,'rb') as file :
        model = pickle.load(file)
    pred_number = model.predict([number])   
    return pred_number



@app.route("/",methods=['GET','POST'])
def Home():
    pred = 0 
    if request.method == 'POST':
        num = request.get_button["btn-1"]   
        return render_template("index.html",pred_number = pred )

if __name__ == '__main__' :
    app.run()