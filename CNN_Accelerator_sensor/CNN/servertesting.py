from flask import Flask, request, jsonify
import random
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas import DataFrame
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
        
        f = request.files["abc"]
        content = f.read()
        now = time.strftime("%Y%m%d-%H%M%S") 
        f2 = open(test_dir+ now +".wav","wb")
        f2.write(content[:])
        #wavio.write('/Users/jason/Thesis/Master_project/SVM_microphone/', my_np_array, fs, sampwidth=2)

        data = request.form['data'].replace('[','').replace(']','')
        print(type(data))
    
    
    qq = data.split(';')[:-1] 
   # print(type(qq))
    #print(len(qq))
    #print(qq)
    df = DataFrame (qq,columns =['values'])
    df[['x-axis', 'y-axis', 'z-axis','timestamp']] = df['values'].str.split(',',expand = True)
    df.drop('values',axis=1)
    #print(df)
    
    #realdata
    realdata=df
    realdata['x-axis'] = realdata['x-axis'].astype(float)
    realdata['y-axis'] = realdata['y-axis'].astype(float)
    realdata['z-axis'] = realdata['z-axis'].astype(float)
    realdata['x-axis'] = feature_normalize(realdata['x-axis'])
    realdata['y-axis'] = feature_normalize(realdata['y-axis'])
    realdata['z-axis'] = feature_normalize(realdata['z-axis'])
    #reshape data 
    realdata = segment_signal(realdata)
    reshaped_realdata = realdata.reshape(len(realdata), 1, 90, 3)
    
    #cnn machine learning
    session =  tf.compat.v1.Session()
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    saver.restore(session, tf.train.latest_checkpoint('./checkpoint_dir'))
    graph = tf.get_default_graph()
    op_to_restore = graph.get_tensor_by_name
    result = session.run(tf.compat.v1.get_default_graph().get_tensor_by_name("prediction:0"), feed_dict={tf.compat.v1.get_default_graph().get_tensor_by_name("X:0"): reshaped_realdata})
    #print(result)
    prediction=('"busy":{:.2f}, "idel":{:.2f}, "pan":{:.2f}, "rotate":{:.2f}'.format(result[0][0], result[0][1], result[0][2], result[0][3]))
    print(prediction)
    
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
            #print(filepath)
            pred= predict(model=model,data_path=filepath)
            l = print_leaderboard(pred=pred,data_path= train_dir)
            os.remove(filepath)
            
    #q =('"idle":{:.2f}, "moving":{:.2f}, "rotate":{:.2f}'.format(l[0][0], l[0][1], l[0][2]))
    q= '"idle"' +':'+ l.get(0)+ ', '+'"moving"' +':'+ l.get(1)   +', '+ '"rotate"'+':'+l.get(2) +', '+ '"pan"'+':'+l.get(2)
    print(q)
    return "{"+q+"}"
    

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
    
   
    