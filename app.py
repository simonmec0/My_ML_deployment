# import Flask
import numpy as np
from flask import Flask, request, render_template
import pickle
from keras.models import load_model

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model.pkl','rb'))
#model = load_model("loan_status_predict_model.h5")
@app.route("/") #home
def home():
    return render_template('index.html')

    
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendring results in HTML GUI
    '''
    #save our features(that we get it from the form) in a list and convert them from a string to a int
    float_features =[float(x) for x in request.form.values()]
    final_features = np.reshape(np.array(float_features),(1,8))
    prediction = model.predict(final_features)

    
    output = round(prediction[0][0],2)
    return render_template('index.html', prediction_text='The loan is good: {} '.format(output))
    
if __name__=='__main__':
    app.run(5005,debug=True)
