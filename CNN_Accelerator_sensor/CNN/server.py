from flask import Flask, request, jsonify
import random
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas import DataFrame

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/',methods=["POST"])
def hello():
    if request.method == 'POST':

        data = request.form['data'].replace('[','').replace(']','')
        #print(data)
        #print(type(data))
        
    qq = data.split(';')[:-1] 
   # print(type(qq))
    #print(len(qq))
    #print(qq)
    df = DataFrame (qq,columns =['values'])
    df[['x-axis', 'y-axis', 'z-axis','timestamp']] = df['values'].str.split(',',expand = True)
    df.drop('values',axis=1)
    #print(df)
    
    
    realdata=df
    realdata['x-axis'] = realdata['x-axis'].astype(float)
    realdata['y-axis'] = realdata['y-axis'].astype(float)
    realdata['z-axis'] = realdata['z-axis'].astype(float)
    realdata['x-axis'] = feature_normalize(realdata['x-axis'])
    realdata['y-axis'] = feature_normalize(realdata['y-axis'])
    realdata['z-axis'] = feature_normalize(realdata['z-axis'])
        
    realdata = segment_signal(realdata)
    reshaped_realdata = realdata.reshape(len(realdata), 1, 90, 3)
    
            
           
    session =  tf.compat.v1.Session()
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    saver.restore(session, tf.train.latest_checkpoint('./checkpoint_dir'))
    graph = tf.get_default_graph()
    op_to_restore = graph.get_tensor_by_name
    result = session.run(tf.compat.v1.get_default_graph().get_tensor_by_name("prediction:0"), feed_dict={tf.compat.v1.get_default_graph().get_tensor_by_name("X:0"): reshaped_realdata})
    #print(result)
    prediction=('"right2":{:.2f}, "right1":{:.2f}, "middle":{:.2f}, "left1":{:.2f},"left2":{:.2f}'.format(result[0][0], result[0][1], result[0][2], result[0][3],result[0][4]))
    print(prediction)
    
    return "{"+prediction+"}"
    

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
    app.run(host='0.0.0.0')
    
   
    