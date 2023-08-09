#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from sklearn.metrics import pairwise


# 1)Set up a function that updates a running average of background values in a Region Of Interest                                          
# 2)This will allow us to detect new object(fingers of hand in the Region Of Interest)
# 
# **STRATEGY FOR COUNTING FINGERS**
# 
# 1)Grab a region of interest.
# 
# 2)Calculate running average background value for 60 frames in the video.
# 
# 3)Once average value is found, the hand can enter the ROI.
# 
# 4)Once hand enters, detect change and apply thresholding.
# 
# 5)Use a Convex Hull to draw a polygon around the hand.
# 
# 6)Calculate center of the hand,calculate center against angle of outer points to infer finger count.
# 
# 

# In[3]:


#global variables
background=None
accumulated_weight=0.5 #halfway point between 0 and 1

#initialize region of interest
roi_top=20
roi_bottom=300
roi_right=300
roi_left=600


# 1)Accumulated weight is used to create a running average of frames over time.
# 
# 2)Employed for background subtraction in video processing tasks, where you want to separate the foreground objects from the background in a video stream.
# 
# 3)accumulated_weight-parameter determines how much the current frame should contribute to the running average. It's a value between 0 and 1, where 0 means the current frame has no effect on the average (essentially no change), and 1 means the current frame completely replaces the previous average (sudden change).

# In[4]:


def calc_accum_avg(frame,accumulated_weight):
    
    global background
    
    #if the background is None, set the background as the frame
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    #else, take the running average of frames if background already initialized
    cv2.accumulateWeighted(frame,background,accumulated_weight)
    



# **SEGMENTATION**
# 
# Use thresholding to grab hand from region of interest and distinguish it's contours.
# 
# 

# In[5]:


def segment(frame, threshold_min=25):
    
    diff = cv2.absdiff(background.astype('uint8'), frame)
    
    #apply threshold on the hand
    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)
    
    #draw contours to the thresholded hand
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    else:
        #maximum of contours applied to hand segment
        hand_segment = max(contours, key=cv2.contourArea) #key to remove noise
        return (thresholded, hand_segment)


# **FINGER COUNTING WITH CONVEX HULL**
# 
# 1)Draws a polygon around the external points in the frame
# 
# 2)Account for lines from the wrist.
# 
# 3)Calculate most extreme points-top,bottom,left and right
# 
# 4)Calculate intersection of extreme points and estimate center of hand.
# 
# 5)Calculate distance of the point furthest away from the center.
# 
# 6)Using the ratio of distance, we create a circle.
# 
# 7)Points outside the circle are extended fingers.

# In[6]:


def count_fingers(thresholded, hand_segment):
    conv_hull = cv2.convexHull(hand_segment)
    
    #calculate extreme points
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    #calculate center
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    #calculate distance between extreme points and center
    distance = pairwise.euclidean_distances(np.array([[cX, cY]]), Y=np.array([left, right, top, bottom]))[0]
    
    max_distance = distance.max()
    
    #calculate the radius based on a ratio of max distance
    radius = int(0.9 * max_distance)
    circumference = int(2 * np.pi * radius)
    
    circular_roi = np.zeros(thresholded.shape[:2], dtype='uint8')
    
    #draw a circle
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    
    #compute external contours on external roi
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #count fingers
    count = 0
    
    for cnt in contours:
        #bounding box on the contours
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        #calculate points that are out of wrist-not wrist points
        out_of_wrist = (cY + (cY * 0.25)) > (y + h)
        #calculate points within limits
        limit_points = ((circumference * 0.25) > cnt.shape[0])
        
        #if they both fall under the category, count variable incremented to one and it is a finger
        if out_of_wrist and limit_points:
            count += 1
            
    return count


# **BRING ALL FUNCTIONS TOGETHER, CONNECT TO CAMERA AND COUNT OUR FINGERS**

# In[7]:


cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    
    ret, frame = cam.read()
    
    frame_copy = frame.copy()
    
    #get region of interest
    roi = frame[roi_top:roi_bottom,roi_right:roi_left]
    
    #convert to grayscale and apply blurring
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    #if the number of frames-per-second is less than 60,keep applying accumulated weight
    if num_frames < 60:
        calc_accum_avg(gray,accumulated_weight)
        
        #if it stops at 59, load error message
        if num_frames <= 59:
            cv2.putText(frame_copy,'WAIT. GETTING BACKGROUND',(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('Finger Count',frame_copy)
    else:
        #segment hand
        hand = segment(gray)
        
        if hand is not None:
            #thresholded hand
            thresholded , hand_segment = hand
            
            # DRAWS CONTOURS AROUND REAL HAND IN LIVE STREAM
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(255,0,0),5)
            #count fingers function
            fingers = count_fingers(thresholded,hand_segment)
            #display number of fingers
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            cv2.imshow('Thresholded',thresholded)
            
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
    
    num_frames += 1
    
    #display finger count on frame copy
    cv2.imshow('Finger Count',frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
    


# In[ ]:





# In[ ]:




