#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:26:09 2020

@author: jason
"""

import pyaudio
import wave
import threading
import time

class audiorecording:
    
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 5


   
    def record_every_five_seconds(self):
        starttime = time.time()
        while True:   
            try:
                now = time.strftime("%Y%m%d-%H%M%S") 
                filename = "/Users/jason/pyAudioClassification-master/example/Testing/"+now+r"report.wav"
                                    
                p = pyaudio.PyAudio()  # Create an interface to PortAudio
                                    
                print('Recording')
                                    
                stream = p.open(format=self.sample_format,
                                channels=self.channels,
                                rate=self.fs,
                                frames_per_buffer=self.chunk,
                                input=True)
                                    
                frames = []  # Initialize array to store frames
                                    
                                    # Store data in chunks for 5 seconds
                for i in range(0, int(self.fs / self.chunk * self.seconds)):
                    data = stream.read(self.chunk)
                    frames.append(data)
                                    
                                    # Stop and close the stream 
                stream.stop_stream()
                stream.close()
                                    # Terminate the PortAudio interface
                p.terminate()
                                    
                                    #print('Finished recording')
                                    
                                    # Save the recorded data as a WAV file
                wf = wave.open(filename, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.sample_format))
                wf.setframerate(self.fs)
                wf.writeframes(b''.join(frames))
                wf.close()
            except KeyboardInterrupt as e:
                break
                    #使用cmd+c退出
                    

AR = audiorecording()
AR.record_every_five_seconds()       xy 