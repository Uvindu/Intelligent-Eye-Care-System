

from skimage.draw import line
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_eye_line(eye_landmarks):
    l0, l1, l2, l3, l4, l5 = eye_landmarks.astype('int')
    A= list(((np.array(l1) + np.array(l5))/2).astype('int'))
    B= list(((np.array(l2) + np.array(l4))/2).astype('int'))

    
    line_ = list(zip(*line(*l0, *A))) + list(zip(*line(*A, *B))) + list(zip(*line(*B, *l3)))

    return line_

def get_frp(image_gray, landmarks, return_img= False, check_errors= False):
    if check_errors==True:
      for point in landmarks:
          point= tuple(map(int, point))
          image_gray = cv2.circle(image_gray, point, 10, (0), -1)
      plt.imshow(image_gray, cmap='gray')
      plt.title('get frp start')
      plt.show() 

    min_pixel_color= 255
    max_pixel_color=0

    eye_line = get_eye_line(landmarks)

    for x,y in eye_line:
        #print(image_gray[y, x])
        gray_color = image_gray[y, x]

        if gray_color>max_pixel_color:
            max_pixel_color=gray_color
            y_max, x_max= [y, x]
        if gray_color<min_pixel_color:
            min_pixel_color=gray_color
            y_min, x_min= [y, x]

    for x,y in eye_line:
        image_gray[y, x]=255
    try:
      image_gray[y_min-2:y_min+2, x_min-2:x_min+2]=0
      image_gray[y_max-2:y_max+2, x_max-2:x_max+2]=0
    except:
      print(f'detected face is incorrect !!! : make check errors= True to see | currect check_errors : {check_errors}')
      if check_errors==True:
        for point in eye_line:
          image_gray = cv2.circle(image_gray, point, 10, (0), -1)
        for point in landmarks:
          point= tuple(map(int, point))
          image_gray = cv2.circle(image_gray, point, 10, (0), -1)
        plt.imshow(image_gray, cmap='gray')
        plt.title('get_frp : detected face incorrect !!!')
        plt.show()

        print('max, min pixel colors : ', max_pixel_color, min_pixel_color)

      else:pass

    frp= (max_pixel_color+0.1)/(min_pixel_color+0.1)
    if return_img:return image_gray, frp
    else:return frp

def img2eye(image_gray, eye_landmark, landmark=None, resize= 28, check_errors= False):
  eye_middle = list(map(int, np.mean(eye_landmark, axis=0)))
  eye_width = int((landmark[9][1] - landmark[28][1])/5)
  eye_image = image_gray[eye_middle[1]-eye_width: eye_middle[1]+eye_width, eye_middle[0]-eye_width: eye_middle[0]+eye_width]
  return cv2.resize(eye_image, (resize, resize))