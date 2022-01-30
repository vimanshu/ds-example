from flask import Flask,request,url_for,render_template
import pandas as pd
import pickle
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# load the pickle file created in example.py
model = pickle.load(open('example_weights_knn.pkl', 'rb'))

@app.route('/')
def use_template():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    input_one = request.form['1']
    input_two = request.form['2']
    input_three = request.form['3']
    input_four = request.form['4']
    input_five = request.form['5']
    input_six = request.form['6']
    input_seven = request.form['7']
    input_eight = request.form['8']

    setup_df = pd.DataFrame(pd.Series([input_one,input_two,input_three,input_four,input_five,input_six,input_seven,input_eight]))
    diabetes_predict = model.predict_proba(setup_df)
    output = diabetes_predict[0][1]
    output = str(float(output)*100)+'%'
    if output >str(50):
        return render_template('result.html',pred = f'You have the following chances of diabetes based on our model.\nProbablity: {output}')
    else:
        return render_template('result.html', pred= f'You have the low chances of diabetes which is currently considered safe [For any medical advice please consult a doctor]\nProbability:{output}')

if __name__=='__main__':
    app.run(debug=True)

