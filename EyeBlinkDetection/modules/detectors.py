import urllib.request as urlreq
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_detectors(detector_name = 'haarcascade'):
  if detector_name == 'haarcascade':
    # https://github.com/Danotsonof/facial-landmark-detection
    haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascade = "haarcascade_frontalface_alt2.xml"
    urlreq.urlretrieve(haarcascade_url, haarcascade)
    detector = cv2.CascadeClassifier(haarcascade)

    LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
    LBFmodel = "LFBmodel.yaml"
    urlreq.urlretrieve(LBFmodel_url, LBFmodel)
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)

    return detector, landmark_detector
  else:
    raise NotImplementedError