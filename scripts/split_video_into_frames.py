# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:27:31 2018

@author: sss
"""

import cv2
import numpy as np
import os
import time

  
time_start = time.time()

# Playing video from file:    
cap = cv2.VideoCapture('./videos/test.mkv')

# Find the number of frames
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print ("Number of frames: ", video_length)

num = 525
count = 0

while cap.isOpened():
    
    # Extract the frame
    ret, frame = cap.read()
    
    if not ret: break

    if(count % 240 == 0): 
        # Saves image of the current frame in jpg file
        num += 1
        name = './slide/slide_' + str(num) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
    
    # To stop duplicate images
    count += 1
        
        
time_end = time.time()           
# Release the feed
cap.release()
# Print stats
print ("It took %d seconds for conversion." % (time_end-time_start))

