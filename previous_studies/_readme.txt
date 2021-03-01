The GazeTracking algorythm has basically several steps..
1. Face Recognition using dlib pretrained model
2. Landmarks Detection by dlib pretrained model
3. Identify the landmarks correspond to eye 
4. capture the eye from the image
5. Find better threshold such that (pupil/eye) ratio => 0.48
6. Extract the eye
7. Get the center of the pupil (by getting contours)

There are several points that we can make changes in the algorythm. 
1. Face recognition/ Landmark detection part
2. Identifying pupil
	they are using noise filtering (cv2.bilateralFilter, cv2.erode) and best threshold (THRESBINARY) in order to extract the pupil