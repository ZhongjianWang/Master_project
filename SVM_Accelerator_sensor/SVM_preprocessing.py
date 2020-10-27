#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 15:45:54 2020

@author: jason
"""
import ast
import math
import numpy as np 
FEATURE = ("mean", "max", "min", "std")
STATUS  = ("Sitting", "Walking", "Upstairs", "Downstairs", "Jogging", "Standing")
def preprocess(file_dir, Seg_granularity):
    gravity_data = []
    with open(file_dir) as f:
        index = 0
        for line in f:
            clear_line = line.strip().lstrip().rstrip(';')
            raw_list = clear_line.split(',') 
            index = index + 1
            if len(raw_list) < 5:
                continue
            status  = raw_list[1] 
            acc_x = float(raw_list[3])
            acc_y = float(raw_list[4])
            print (index)
            acc_z = float(raw_list[5])
            if acc_x == 0 or acc_y == 0 or acc_z == 0:
                continue
            gravity = math.sqrt(math.pow(acc_x, 2)+math.pow(acc_y, 2)+math.pow(acc_z, 2))
            gravity_tuple = {"gravity": gravity, "status": status}
            gravity_data.append(gravity_tuple)
    # split data sample of gravity
    splited_data = []
    cur_cluster  = []
    counter      = 0
    last_status  = gravity_data[0]["status"]
    for gravity_tuple in gravity_data:
        if not (counter < Seg_granularity and gravity_tuple["status"] == last_status):
            seg_data = {"status": last_status, "values": cur_cluster}
            # print seg_data
            splited_data.append(seg_data)
            cur_cluster = []
            counter = 0
        cur_cluster.append(gravity_tuple["gravity"])
        last_status = gravity_tuple["status"]
        counter += 1
    # compute statistics of gravity data
    statistics_data = []
    for seg_data in splited_data:
        np_values = np.array(seg_data.pop("values"))
        seg_data["max"]  = np.amax(np_values)
        seg_data["min"]  = np.amin(np_values)
        seg_data["std"]  = np.std(np_values)
        seg_data["mean"] = np.mean(np_values)
        statistics_data.append(seg_data)
    # write statistics result into a file in format of LibSVM
    with open("./WISDM_ar_v1.1/WISDM_ar_v1.1_raw_svm.txt", "a") as the_file:
        for seg_data in statistics_data:
            row = str(STATUS.index(seg_data["status"])) + " " + \
                  str(FEATURE.index("mean")) + ":" + str(seg_data["mean"]) + " " + \
                  str(FEATURE.index("max")) + ":" + str(seg_data["max"]) + " " + \
                  str(FEATURE.index("min")) + ":" + str(seg_data["min"]) + " " + \
                  str(FEATURE.index("std")) + ":" + str(seg_data["std"]) + "\n"
            # print row
            the_file.write(row)        
if __name__ == "__main__":
    preprocess("./WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt", 100)
    pass  