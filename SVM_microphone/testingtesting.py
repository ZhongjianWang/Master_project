#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:01:03 2020

@author: jason
"""
import pyaudio
import wave
import threading
import time
import numpy as np
import os
from pyaudioclassification import feature_extraction, train, predict, print_leaderboard #在init里面把所有function都写入class里，调用方法:class名字().feature_extraction(data_path=)

test_dir = 'testing/' 
train_dir = 'data/'




def my_func(x):
    return x

if __name__ == '__main__':
    # Put that needs to run here
# step 1: preprocessing
    if np.DataSource().exists("feat.npy") and np.DataSource().exists("label.npy"):
        features, labels = np.load('feat.npy'), np.load('label.npy')
    else:
        features, labels = feature_extraction('data/')
        np.save('feat.npy', features)
        np.save('label.npy', labels)
    print (features, labels)    
    def process(test_dir):
        while True:
            
            # step 2: training
            if np.DataSource().exists("model.h5"):
                from keras.models import load_model
                model = load_model('model.h5')
            else:
                model = train(features, labels, epochs=100)
                model.save('model.h5')
        
            # step 3: prediction 更改filepath,go through all folder 得到file path，在init文件里面predict function里面更改,可以自动检索文件然后不断运行，达到实时测试的效果
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                        filepath = os.path.join(root, file)
                        print(filepath)
                        pred= predict(model=model,data_path=filepath)
                        print_leaderboard(pred=pred,data_path=train_dir)
                        os.remove(filepath)
                        #option1: delete it search for the command in python to delete file 
                        
                        #option2: os.path.getmtime()how to check the folder is updated 
def call_upper_method_and_break_if_keyboard_used():
        while True:
            try:
                process()
            except KeyboardInterrupt as e:
                #使用cmd+c退出
                break

process(test_dir=test_dir)
