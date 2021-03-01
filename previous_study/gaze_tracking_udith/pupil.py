import numpy as np
import cv2
m=6
def high(img):
    img = cv2.resize(img,(int(img.shape[1]*m),int(img.shape[0]*m)))
    dst = cv2.fastNlMeansDenoising(img,None,200.0, 7, 21)
    #dst2 = cv.fastNlMeansDenoising(dst,None,20.0,7,21)
    
    #dst3 = cv.fastNlMeansDenoising(dst2,None,50,7,21)
    '''
    dst4 = cv.fastNlMeansDenoisingColored(dst3,None,10,10,7,21)
    dst5 = cv.fastNlMeansDenoisingColored(dst4,None,10,10,7,21)
    dst6 = cv.fastNlMeansDenoisingColored(dst5,None,10,10,7,21)
    dst7 = cv.fastNlMeansDenoisingColored(dst6,None,10,10,7,21)
    dst8 = cv.fastNlMeansDenoisingColored(dst7,None,10,10,7,21)
    dst9 = cv.fastNlMeansDenoisingColored(dst8,None,10,10,7,21)
    dst10 = cv.fastNlMeansDenoisingColored(dst9,None,10,10,7,21)'''

    return dst #RGB

def blob(im):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 270  # The dot in 20pt font has area of about 30
    params.filterByColor = 1
    #params.blobColor = 255
    params.filterByCircularity = 0
    params.filterByConvexity = 0
    params.filterByInertia = 0
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im)
    try:
        return(keypoints[0].pt[0]//m,keypoints[0].pt[0]//m)
    except:
        return(('nun','nun'))



    

class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        self.x,self.y=np.array(blob(high(eye_frame)),dtype='uint8')
        
