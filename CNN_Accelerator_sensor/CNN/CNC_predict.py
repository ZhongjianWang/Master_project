#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:04:38 2020

@author: jason
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 17:55:50 2020

@author: jason
"""
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def read_data(file_path):
    column_names = ['activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header=None, names=column_names)
    data= data[1:]
    print(type(data))
    print(data)
    return data

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
    labels = np.empty((0))
    print(len(data['timestamp']))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if (len(data['timestamp'][start:end]) == window_size):
            print((start, end))
            segments = np.vstack([segments, np.dstack([x, y, z])])
    return segments, labels

realdata = read_data('./WISDM_ar_v1.1/CNC_testdata1.csv')
realdata['x-axis'] = realdata['x-axis'].astype(float)
realdata['y-axis'] = realdata['y-axis'].astype(float)
realdata['z-axis'] = realdata['z-axis'].astype(float)
realdata['x-axis'] = feature_normalize(realdata['x-axis'])
realdata['y-axis'] = feature_normalize(realdata['y-axis'])
realdata['z-axis'] = feature_normalize(realdata['z-axis'])

realdata, reallabel = segment_signal(realdata)
reshaped_realdata = realdata.reshape(len(realdata), 1, 90, 3)
print(len(realdata))
print(reshaped_realdata.shape)

session=tf.Session()
#先加载图和参数变量
saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
saver.restore(session, tf.train.latest_checkpoint('./checkpoint_dir'))


# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()

#接下来，访问你想要执行的op
op_to_restore = graph.get_tensor_by_name

result_cnn= session.run(op_to_restore("prediction:0"), feed_dict={op_to_restore("X:0"): reshaped_realdata})
print(result_cnn)
#print("busy:{:.2f}, idel:{:.2f}, pan:{:.2f}, rotate:{:.2f}".format(result_cnn[0][0], result_cnn[0][1], result_cnn[0][2], result_cnn[0][3])) 
session.close()



    