"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    try:
        gaze.refresh(frame)
    except:pass
    try:
        print(gaze.calibration.iris_size(gaze.eye_right.pupil.iris_frame))
        #print(gaze.calibration.find_best_threshold(gaze.eye_left.frame))
        #print(gaze.calibration.thresholds_left)
    except:
        print('pupil detection failed')
        try:
            landmarks = self._predictor(frame, faces[0])
        except:
            print("landmarks detection failed")
            try:
                faces= self._face_detector(frame)
            except:print("face_detection failed")
    frame = gaze.annotated_frame()
    text = ""
    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
