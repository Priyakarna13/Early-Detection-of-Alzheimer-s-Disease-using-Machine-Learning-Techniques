import pickle
import pandas as pd


def pred(data):
    model = pickle.load(open('finalized_model.sav', 'rb'))
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
        
    data_test = pd.DataFrame(data)  #M/F,Age,EDUC,SES,MMSE,eTIV,nWBV,ASF
    test = scaler.transform(data_test)
    PredictedOutput = model.predict(test)
    return PredictedOutput

def class_pred(x):
    if x[0] == 1:
        return "Category: Demented"
    else:
        return "Category: Non Demented"
    
def ask():
    l = []
    
    l.append(int(input("Gender [1:male, 0:female]:")))
    l.append(int(input("Age:")))
    l.append(int(input("EDUC:")))
    l.append(float(input("SES:")))
    l.append(float(input("MMSE:")))
    l.append(int(input("eTIV:")))
    l.append(float(input("nWBV:")))
    l.append(float(input("ASF:")))
    
    return l

def check():
    return class_pred(ask())