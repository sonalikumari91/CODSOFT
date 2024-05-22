import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

## import DecisionTreeRegressor model and standard scaler pickle
Logistic_Regression_model=pickle.load(open(r'C:\Users\Gaurav\OneDrive\Desktop\codsoft\model\logistic.pkl','rb'))
standard_scaler=pickle.load(open(r'C:\Users\Gaurav\OneDrive\Desktop\codsoft\model\scaler.pkl','rb'))

# Route for home page
@app.route("/")
def index():
    return render_template('abcd.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        sex=float(request.form.get('sex'))
        Pclass=float(request.form.get('Pclass'))
        siblings=float(request.form.get('siblings'))
        age=float(request.form.get('age'))
        par_chi=float(request.form.get('par_chi'))
        Embarked=float(request.form.get('Embarked'))
        

        new_data_scaled=standard_scaler.transform([['sex','Pclass','siblings','age','par_chi','Embarked',]])
        result=Logistic_Regression_model.predict(new_data_scaled)
        return render_template('index.html',result=result[0])
    else:
        return render_template('abcd.html')

if __name__ == '__main__':
    app.run(debug= True)


