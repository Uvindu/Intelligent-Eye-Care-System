import cv2
import numpy as np
import matplotlib.pyplot as plt

def image2features(image_gray, feature_func, faces, landmark_detector=None, check_errors=False): #features can be eye images/ frp-like features
  landmark = landmark_detector.fit(image_gray, faces)[1][0][0] #landmarks of first person
  
  
  eye1_landmark= landmark[36:42]
  eye2_landmark= landmark[42:48]

  if check_errors==True:
    image_gray_show= image_gray.copy()  
    for point in landmark:
        point= tuple(map(int, point))
        image_gray_show = cv2.circle(image_gray_show, point, 10, (0), -1)
    plt.imshow(image_gray_show, cmap='gray')
    plt.title('image2features :start')
    plt.show()   
    

  features= feature_func(image_gray, eye1_landmark, eye2_landmark, landmark, check_errors) # return 2 features for both eyes
  return features

def get_features_for_sample(dataset_filedirvslabel, sample_id, feature_func, detector=None, landmark_detector=None, check_errors= False):
  vid_frame_dirs, label = dataset_filedirvslabel[sample_id]

  features_eye1, features_eye2=[], []
  for img_dir in vid_frame_dirs:
    image = cv2.imread(img_dir)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(image_gray)

    if len(faces)!=0:
      feature_eye1, feature_eye2 = image2features(image_gray, feature_func, faces, landmark_detector, check_errors)

      features_eye1.append(feature_eye1)
      features_eye2.append(feature_eye2)

    else:
      landmark= 'no face'
      print(f"no faces detected : {img_dir}")
      if check_errors:
        plt.imshow(image_gray)
        plt.title('get_features_for_sample: No faces detected !!!, return None, None')
        plt.show()
      return None, None
  return np.array([features_eye1, features_eye2]), label