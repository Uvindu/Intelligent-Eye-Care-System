This repo contains the optimized gaze detection using openSource library https://github.com/antoinelame/GazeTracking
BLOB_DETECTION algorythm is used for detecting gaze.


### About the notebooks:

1. **dlib_models.ipynb ? RUN**
    - Some experiments with dlib face_detectors
    - extracting points, faces, etc
2. **test.ipynb ? NO NEED TO RUN**
    - Pipelines for obtaining iris frames from raw images ()
    - Testing the impact of iris_frame with different thresholds (using official libraries)
    - Store the landmarks, frame, points as common variables
3. **detect._iris.ipynb -** Tried methods (not worked all with the knowledge I had), Therefore all the results of those methods can be improved with the knowledge of ayyalas)- ? **NO NEED TO RUN**
    1. filters: sobel, bilateral
    2. Hough detection- good
    3. watershed- somewhat good
    4. grabcut
    5. blob detection
    6. some others...
4. **blob_detection.ipynb -**
    - For find center of pupils, they have used "contour detection and finding the center" ****
    - Here, I have tried blurring + blob detection for find pupil/ center
    - blurring ? was used because with the sharpness of the image/ noise of the image, blob detection was not worked well (As I remembered)
    - Checked the color histograms of eyes
5. **image_processing.ipynb (run test.ipynb before this) ? NO NEED TO RUN**
    - experiments with image_processing script (their iris detection from eyes algorithm)
    - Impact of thresholds
6. **_isolate.ipynb (run test.ipynb before this) ? NO NEED TO RUN**
    - experiments with **_isolate** function
    - isolate the eye from the whole frame (check)
7. **realtime_eye_frame_only.ipynb ? RUN**
    - Showing real time eyes
8. **realtime_iris_frame_only.ipynb ? RUN**
    - Showing real time iris
9. **filtered_pupil_centerpoint_realitime.ipynb (IMPORTANT) ? RUN**
    - Previous algorithms were changed a bit/ calibrations added
    - This was the best results when I stopped working on this
10. **example.ipynb (NO NEED TO GO THROUGH)**
    - experiments with classes of repository
        - eg: gaze.eye_right.pupil.iris_frame, etc
