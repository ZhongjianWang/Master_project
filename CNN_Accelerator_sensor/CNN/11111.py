from flask import Flask, request, jsonify
import random
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas import DataFrame
import joblib
#import _main_
from keras.models import load_model
from pyaudioclassification import feature_extraction, train, predict, print_leaderboard
import os
fs = 44100
import requests
from pyaudioclassification.feat_extract import parse_audio_files, parse_audio_file
import time
#import wavio
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
test_dir = '/Users/jason/Thesis/Master_project/SVM_microphone/Testing/' 
train_dir = '/Users/jason/Thesis/Master_project/SVM_microphone/data/'


@app.route('/',methods=["POST"])
def hello():
    if request.method == 'POST':
        
        print("receive data for phone")
        # audio data
        f = request.files["abc"]
        content = f.read()
        now = time.strftime("%Y%m%d-%H%M%S") 
        f2 = open(test_dir+ now +".wav","wb")
        f2.write(content[:])
         
        # Magnetic data
        data = request.form['data'].replace('[','').replace(']','')
      
        qq = data.split(';')[:-1] 
        df = DataFrame (qq,columns =['values'])
        df[['x-axis', 'y-axis', 'z-axis','timestamp']] = df['values'].str.split(',',expand = True)
        df.drop('values',axis=1)
        magneticdata=df

    
    # Action recognize
    c= Audio_process_function()
    
    return sensor_active(magneticdata,c.get(0),c.get(2))
    
    # Direction recognize
    



# activate function
def sensor_active(magdata,a,b):
    magdata_x = magdata['x-axis'].astype(float)
    
    if abs(magdata_x[1] - magdata_x[90]) < 3 :
        print("Not moving")
        
        audio_result= '"idle"' +':'+ a+ ', '+'"left"' +':'+ '0'   +', '+ '"rotate"'+':'+b +', '+ '"right"'+':'+'0'
        return "{"+audio_result+"}"
        # No process function
        
    else :
        print("moving")
        # Process function
        result = 0
        for i in range(0, 90):
            if (magdata_x[89] - magdata_x[i]) >= 0:
                result = result + 1
            else:
                result = result - 1
        
        if Mag_process_function(result) == 0:
            right= '"idle"' +':'+ a+ ', '+'"left"' +':'+ '0'   +', '+ '"rotate"'+':'+b +', '+ '"right"'+':'+'100'
            return "{"+right+"}"
        else:
            left='"idle"' +':'+ a+ ', '+'"left"' +':'+ '100'   +', '+ '"rotate"'+':'+b +', '+ '"right"'+':'+'0'
            return "{"+left+"}"

        

# Audio process function
def Audio_process_function():
    #svm machine learning
    model = load_model('/Users/jason/Thesis/Master_project/SVM_microphone/model.h5')
    #data_path= test_dir
    features, labels = np.load('feat.npy'), np.load('label.npy')
    #x_data = parse_audio_file(data_path)
    #X_train = np.expand_dims(x_data, axis=2)
    #pred = model.predict(X_train)
    l = {}
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            filepath = os.path.join(root, file)
            print(filepath)
            pred= predict(model=model,data_path=filepath)
            l = print_leaderboard(pred=pred,data_path = train_dir)
            os.remove(filepath)
    print(l.get(0))
    q= '"idle"' +':'+ l.get(0)+ ', '+'"moving"' +':'+ l.get(1)   +', '+ '"rotate"'+':'+l.get(2) +', '+ '"pan"'+':'+l.get(2)
    print(q)
    
    return l
    #return "{"+q+"}"
    
def Mag_process_function(result):
    modelpath = 'finalized_model.pkl'
    loaded_model = joblib.load(modelpath)
    loaded_scaler = joblib.load('finalized_scaler.pkl')
    #Predicting the Test set results  
    XR = np.arange(1)
    XR[0] = result
    print("XR: ")
    print(XR)
    XR = XR.reshape(-1, 1)
    print("XR_reshaped: ")
    print(XR)
   # from sklearn.preprocessing import StandardScaler
    #sc = StandardScaler()
    XR = loaded_scaler.transform(XR)
    print("XR_scaled: ")
    print(XR)
    y_pred = loaded_model.predict(XR)
    if y_pred == 0:
        print("Right")
        return 0
    else:
        print("Left")
        return 1
    
    
    
    
    
  

def feature_normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma 

def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start + size
        start += int(size / 2)
        
def segment_signal(data, window_size=90):
    segments = np.empty((0, window_size, 3))
    #print(len(data['timestamp']))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(data['timestamp'][start:end]) == window_size):
            #print((start, end))
            segments = np.vstack([segments, np.dstack([x, y, z])])
    return segments



if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False)
    
       