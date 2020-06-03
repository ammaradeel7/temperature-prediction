from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler


from csv import writer

def appendTofile(file_name, row):
    with open(file_name, "a") as myfile:
        myfile.write(row)


def listToString(s):  
    
    # initialize an empty string 
    str1 = "\n54555,"  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele
        str1 += ','   
       
    return str1[:-1]  
        



def scaleNumericData(X,numerical):
  scaler = MinMaxScaler()
  X[numerical] = scaler.fit_transform(X[numerical])
  return X

def encodeCategorical(X):
  X = pd.get_dummies(X, drop_first=True)
  return X


def readAndPreprocess():
  df = pd.read_csv('standard.csv')  
  df =df.drop(df.columns[0], axis=1)
  categorical = [var for var in df.columns if df[var].dtype=='O']
  numerical = [var for var in df.columns if df[var].dtype!='O']
  df = scaleNumericData(df,numerical)
  df = encodeCategorical(df)
  return df


model = 0

def loadModel():
    global model
    model = pickle.load(open('model.pkl','rb'))
    





app = Flask(__name__,template_folder='template')



@app.route('/')
def hello_world():
  loadModel()
  return render_template("tempraturePrediction.html")




@app.route('/predict',methods=['POST','GET'])
def predict():

    
    
    int_features=[x for x in request.form.values()]
    
    appendTofile('standard.csv',listToString(int_features))
    df = readAndPreprocess()
    
  
    
 
  #  print(df.tail(2), file=sys.stderr)
    pred = model.predict(df) 
    length = len(pred)
    
    return "The predicted Average temprature is " + str(pred[length-1])



if __name__ == '__main__':
    app.run(debug=True)