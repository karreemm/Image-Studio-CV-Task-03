import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from classes.featureExtraxtion import FeatureExtraction
from classes.sift import SIFT
from copy import deepcopy
from classes.image import Image

class Controller():
    def __init__(self , labels ):
        self.input_image = Image() 
        self.output_image = Image()
        self.input_image_matching_1 = Image()
        self.input_image_matching_2 = Image()
        self.labels = labels
        self.sift1 = SIFT(sigma=1.6, s=3, num_octaves=4)
        self.sift2 = SIFT(sigma=1.6, s=3, num_octaves=4)
        self.feature_extraction = FeatureExtraction()  
    
    def browse_input_image(self):
        self.input_image.select_image()
        self.output_image= deepcopy(self.input_image) 
        if self.input_image.input_image is not None:
            # Convert the image to QPixmap and display it in the input frame
            qpixmap = self.numpy_to_qpixmap(self.input_image.input_image)
            self.labels[0].setPixmap(qpixmap)
            self.labels[1].setPixmap(qpixmap)
            self.labels[2].setPixmap(qpixmap)
            self.labels[3].setPixmap(qpixmap)

    def browse_matching_image1(self):
        self.input_image_matching_1.select_image()
        if self.input_image_matching_1.input_image is not None:

            qpixmap_input_image_1 = self.numpy_to_qpixmap(self.input_image_matching_1.input_image)
            self.labels[4].setPixmap(qpixmap_input_image_1)

    def browse_matching_image2(self):
        self.input_image_matching_2.select_image()
        if self.input_image_matching_2.input_image is not None:

            qpixmap_input_image_2 = self.numpy_to_qpixmap(self.input_image_matching_2.input_image)
            self.labels[5].setPixmap(qpixmap_input_image_2)

    def apply_lambda_minus_extraction(self, window_size, threshold, sigma):
        features = self.feature_extraction.extraxt_lamda_minus(self.input_image.input_image, window_size, threshold, sigma)
        self.output_image.input_image = features
        qpixmap_ = self.numpy_to_qpixmap(features)
        self.labels[1].setPixmap(qpixmap_)

    def apply_sift(self, sigma, num_octaves):
        # Apply SIFT feature extraction
        self.sift1.sigma = sigma
        self.sift1.num_octaves = num_octaves
        self.sift1.build_scale_space(self.input_image.input_image)
        extrema1 = self.sift1.detect_extrema()
        keypoints1 = self.sift1.localize_keypoints(extrema1)
        oriented_keypoints = self.sift1.assign_orientations(keypoints1, self.input_image.input_image)
        final_keypoints, descriptors = self.sift1.compute_descriptors(oriented_keypoints, self.input_image.input_image)
        
        result_image = cv2.cvtColor(self.input_image.input_image.copy(), cv2.COLOR_GRAY2BGR)
        
        hues = np.linspace(0, 179, len(final_keypoints)) 
        colors = []
        for hue in hues:

            hsv_color = np.uint8([[[hue, 255, 255]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(bgr_color)
        
        # Draw keypoints on the image with different colors
        for idx, keypoint in enumerate(final_keypoints):
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])  
            color = (int(colors[idx][0]), int(colors[idx][1]), int(colors[idx][2]))
            cv2.circle(result_image, (x, y), 5, color, -1)
        
        # Store the result and convert to QPixmap
        self.output_image.input_image = result_image
        qpixmap__ = self.numpy_to_qpixmap(result_image)
        self.labels[3].setPixmap(qpixmap__)
    
    def match_images(self, method="ssd"):
        self.sift1.build_scale_space(self.input_image_matching_1.input_image)
        extrema1 = self.sift1.detect_extrema()
        keypoints1 = self.sift1.localize_keypoints(extrema1)
        oriented_keypoints1 = self.sift1.assign_orientations(keypoints1, self.input_image_matching_1.input_image)
        final_keypoints1, descriptors1 = self.sift1.compute_descriptors(oriented_keypoints1, self.input_image_matching_1.input_image)
        
        self.sift2.build_scale_space(self.input_image_matching_2.input_image)
        extrema2 = self.sift2.detect_extrema()
        keypoints2 = self.sift2.localize_keypoints(extrema2)
        oriented_keypoints2 = self.sift2.assign_orientations(keypoints2, self.input_image_matching_2.input_image)
        final_keypoints2, descriptors2 = self.sift2.compute_descriptors(oriented_keypoints2, self.input_image_matching_2.input_image)

        matched_image = self.sift1.match_and_visualize(self.input_image_matching_1.input_image, self.input_image_matching_2.input_image, final_keypoints1, descriptors1, final_keypoints2, descriptors2, method)
        qpixmap_ = self.numpy_to_qpixmap(matched_image)
        self.labels[6].setPixmap(qpixmap_)
        
    def numpy_to_qpixmap(self, image_array):
        """Convert NumPy array to QPixmap"""
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        height, width, channels = image_array.shape
        bytes_per_line = channels * width
        qimage = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage)
    
