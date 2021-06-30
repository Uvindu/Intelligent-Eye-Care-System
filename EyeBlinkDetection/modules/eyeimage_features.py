
from skimage.feature import local_binary_pattern
import numpy as np

def eyeimages2lbp(eyeimages_dataset_batch, feature_length= 100): # eyeimages: (10, 2, 28, 28)
    new_dataset = []
    for datapoint in eyeimages_dataset_batch:
        botheye_video, label = datapoint
        
        features_eyes = []
        for eye_video in botheye_video:
            eye_video_features = []
            for eye in eye_video:
              lbp_features= local_binary_pattern(eye, 8*3, 3, 'uniform').flatten()
              lbp_features_resized= np.interp(range(feature_length), range(len(lbp_features)), lbp_features)
              eye_video_features.append(lbp_features_resized)
            
            features_eyes.append(eye_video_features)
        new_dataset.append([np.array(features_eyes), label])
    return new_dataset