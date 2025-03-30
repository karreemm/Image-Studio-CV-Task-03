import cv2
from PyQt5.QtGui import QPixmap , QImage
from classes.featureExtraxtion import FeatureExtraction
from classes.sift import SIFT
from copy import deepcopy

class Controller():
    def __init__(self , input_image , output_image ):
        self.input_image = input_image
        self.output_image = output_image
        self.sift = SIFT()  
        self.feature_extraction = FeatureExtraction()  
    def browse_input_image(self):
        self.input_image.select_image()
       
    
    
        
    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array to QPixmap"""
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
    def apply_lambda_minus_extraxtion(self, threshold, window_size):
        if self.input_image.input_image is not None:
            # Step 1: Apply lambda minus feature extraction
            features = self.feature_extraction.extraxt_lamda_minus(self.input_image.input_image, threshold, window_size)
            
            # Step 2: Draw the features on the output image
            result_image = self.input_image.input_image.copy()
            for feature in features:
                cv2.circle(result_image, (int(feature[0]), int(feature[1])), 5, (0, 255, 0), -1)
            