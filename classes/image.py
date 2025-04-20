import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox

class Image():
    def __init__(self):
        self.input_image = None # kept untouched
        self.output_image = None # the one that will be modified by noise, filters etc.
        self.image_type = None # 'grey' or 'color' image
        self.input_image_fft = None # input image in frequency domain
        self.output_image_fft = None # output image in frequency domain

    def select_image(self):
        '''
        function to select and upload an image 
        '''
        try:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif)", options=options)
            if file_path:
                self.input_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                self.output_image = self.input_image.copy() # a copy of the selected image is made so we can modify it without affecting the original image
                self.update_image_type(self.input_image) # update the selected image type
        except Exception as e:
            QMessageBox.warning(None , "Browse an image" , "Please select an image to browse")
        

    def update_image_type(self, image):
        '''
        function that detects whether image is grey or color (rgb) and updates the image_type attribute
        '''
        # Check if image is grayscale by checking dimensions
        if len(image.shape) == 2:
            self.image_type = 'grey'
            return

        # If image has 3 dimensions, check if it's truly color or just grayscale in BGR format
        r,g,b = cv2.split(image)

        # getting differences between them
        r_g = np.count_nonzero(abs(r-g))
        r_b = np.count_nonzero(abs(r-b))
        g_b = np.count_nonzero(abs(g-b))
        diff_sum = float(r_g+r_b+g_b)

        # finding ratio of diff_sum with respect to size of image
        ratio = diff_sum/image.size
        if ratio > 0.005:
            self.image_type = 'color'
        else:
            self.image_type = 'grey'

