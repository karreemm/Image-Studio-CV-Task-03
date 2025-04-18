import  numpy as np
import cv2
class FeatureExtraction:
    def __init__(self):
        ''' Harris and lambda minus parameters'''
        self.window_size = None
        self.corners = None
        self.H_matrix = None
        self.lambda_minus = None
        
    def extraxt_lamda_minus(self, image, window_size = 3):    
        ''' Extract lambda minus features '''
        
        # Initalization
        self.lambda_minus = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        self.window_size = window_size
        
        # First step --> Calculate gradient using sobel
        self.calculate_gradient(image)
        
        # Second step --> Get H matrix and calculate eigenvalues(lambda minus)
        self.calculate_H_matrix()
        
        # Third step --> Normalization
        self.normalize_lambda_minus()     
           
        return self.corners
        
    
    def calculate_gradient(self, image):
        ''' Calculate the gradients of the image using Sobel '''
        
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        self.Ixx = sobel_x ** 2
        self.Iyy = sobel_y ** 2
        self.Ixy = sobel_x * sobel_y      
        self.Iyx = sobel_y * sobel_x
        
        
    def calculate_H_matrix(self):
        ''' Calculate the Harris matrix '''
        
        self.Ixx = self.pad_image(self.Ixx)
        self.Iyy = self.pad_image(self.Iyy)
        self.Ixy = self.pad_image(self.Ixy)
        height, width = self.Ixx.shape
        half_window = (self.window_size - 1) // 2  

        for x in range(half_window, height - half_window):
            for y in range(half_window, width - half_window):
                
                Ixx_window = self.Ixx[x - half_window : x + half_window + 1, y - half_window : y + half_window + 1]
                Iyy_window = self.Iyy[x - half_window : x + half_window + 1, y - half_window : y + half_window + 1]
                Ixy_window = self.Ixy[x - half_window : x + half_window + 1, y - half_window : y + half_window + 1]

                Sxx = np.sum(Ixx_window)
                Syy = np.sum(Iyy_window)
                Sxy = np.sum(Ixy_window)
                
                H = np.array([[Sxx, Sxy],
                            [Sxy, Syy]])

                self.calculate_eigenvalues(H, y - half_window, x - half_window)

        
    
    def calculate_eigenvalues(self, H_matrix, y, x):
        ''' Calculate the eigenvalues and eigenvectors of the Harris matrix '''
        
        eigenvalues = np.linalg.eigvals(H_matrix)
        self.lambda_minus[y, x] = min(eigenvalues)
    
    def apply_thresholding(self):
        ''' Apply thresholding to choose lambda minus '''
        
        
        self.corners = np.where(self.lambda_minus > self.threshold, 255, 0)
        
    def pad_image(self, image):
        ''' Pad the image to avoid border exceptions '''
        
        height, width = image.shape
        padded_image = np.zeros((height + 2, width + 2), dtype=image.dtype)
        padded_image[1:-1, 1:-1] = image
        return padded_image    
    
    def normalize_lambda_minus(self):
        ''' Normalize the lambda minus matrix between 0 and 255 '''
        
        self.lambda_minus = (self.lambda_minus - np.min(self.lambda_minus)) / (np.max(self.lambda_minus) - np.min(self.lambda_minus)) * 255
        self.corners = self.lambda_minus.astype(np.uint8)  # for 8 bit representation

import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def main():
    # Initialize the feature extraction class
    feature_extractor = FeatureExtraction()

    # Use a file dialog to select an image
    Tk().withdraw()  # Hide the root Tkinter window
    image_path = askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    
    if not image_path:
        print("No image selected. Exiting...")
        return

    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Failed to load the image. Exiting...")
        return

    # Extract features
    threshold = 20  # You can adjust this threshold
    window_size = 3  # You can adjust the window size
    features = feature_extractor.extraxt_lamda_minus(image, window_size=window_size)


    # Display the original image and the lambda_minus matrix
    cv2.imshow("Original Image", image)
    cv2.imshow("Lambda Minus Matrix", features)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()