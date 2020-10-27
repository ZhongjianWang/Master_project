#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:23:55 2020

@author: jason
"""
import pandas as pd
import numpy as np

#transform the data 
'''df=pd.read_csv('./WISDM_ar_v1.1/CNC_data.csv')

df=df[['activity', 'timestamp', 'values']]
df[['x-axis', 'y-axis', 'z-axis']] = df['values'].str.split(',',expand = True)
df['x-axis'] = df['x-axis'].astype(float)
df['y-axis'] = df['y-axis'].astype(float)
df['z-axis'] = df['z-axis'].astype(float)
df= df.drop(['values'], axis=1)
df.to_csv('./WISDM_ar_v1.1/CNC_data_cleaned.csv', index= False )

#read new data

df=pd.read_csv('./WISDM_ar_v1.1/magnetdata.csv')
df.columns = ['timestamp', 'activity', 'x-axis', 'y-axis', 'z-axis']
df=df[['activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']]
df.to_csv('./WISDM_ar_v1.1/magnetdata_cleaned.csv', index= False )




df_test1=df[:900]
df_test2=df[-900:]
df_test1.to_csv('./WISDM_ar_v1.1/CNC_testdata1.csv',index= False)
df_test2.to_csv('./WISDM_ar_v1.1/CNC_testdata2.csv',index= False)


def compare(a,b):
    if a > b:
        return a
    else:
        return b
    
def input_func():
    print("input A:",end = " ")
    a = float(input())
    print("input B:",end = " ")
    b = float(input())
    print(compare(a,b))

    return

def main():
    input_func()

    return


main()

Apple = 100
def fun(a=100,b,c): 
    return a + b + c 

    print(fun())
print('a past =',a )

def argument_test(some_argument, window_size,a=19999):
    print(some_argument)
    print(window_size)
    print(a)

argument_test(4,777)'''


def BMI(name,h,m):
    bmi= m/(h**2)
    print(name,"bmi: ", bmi)
   # print(bmi)
    if bmi < 25:
        return name + " is not overweight"
    else:
        return name + " is overweight"

a= BMI('YK',2,90)
print(a)
b= BMI('YK_sister',1.8,70)
print(b)
c= BMI('YK_brother',2.5,160)
print(c)
    