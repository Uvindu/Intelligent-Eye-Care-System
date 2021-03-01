"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import time
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

t_def_min=4
blink_time_array=[]
mode=1
text='READY TO GO'
start=time.time()
i=-1
while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    
    if gaze.is_blinking():
        i+=1
        print('MODE_'+str(mode)+':',i)
        blink_time_array.append(time.time()-start)
        if len(blink_time_array)>10:
            blink_time_array=blink_time_array[1:]
        if len(blink_time_array)==10:   
            t_def=blink_time_array[-1]-blink_time_array[0]
        else:t_def=5
        if t_def_min>t_def:
            t_def_min=t_def
    if t_def_min<=2:
        i=-1
        if mode==1:
            mode=2
            t_def_min=4
            blink_time_array=[]
            text='MODE_2'
        else:
            mode=1
            t_def_min=4
            blink_time_array=[]
            text='MODE_1'

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
