
from modules.feature_functions_utils import *

def feature_func_eyeimages(image_gray, eye1_landmark, eye2_landmark, all_landmarks, check_errors= False):
  eye1_img = img2eye(image_gray, eye1_landmark, all_landmarks, resize= 28, check_errors= check_errors)
  eye2_img = img2eye(image_gray, eye2_landmark, all_landmarks, resize= 28, check_errors= check_errors)

  return np.array([eye1_img, eye2_img])

def feature_func_frp(image_gray, eye1_landmark, eye2_landmark, all_landmarks, check_errors= False):
  eye1_frp = get_frp(image_gray, eye1_landmark, return_img= False, check_errors= check_errors)
  eye2_frp = get_frp(image_gray, eye2_landmark, return_img= False, check_errors= check_errors)

  return np.array([eye1_frp, eye2_frp])