import  numpy as np
import cv2
class FeatureExtraction:
    def __init__(self):
        ''' Harris and lambda minus parameters'''
        self.window_size = None
        self.threshold = 0
        self.corners = None
        self.H_matrix = None
        self.lambda_minus = None
        self.harris_response = None
        self.sigma = 1.0 
        self.window_kernel = None
        self.method = "Harris"
        
    def extraxt_lamda_minus(self, image, method, window_size = 3, threshold = 0.1, sigma = 0):    
        ''' Extract lambda minus features '''
        # Initalization
        self.lambda_minus = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        self.harris_response = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        self.window_size = window_size
        self.threshold = threshold
        self.sigma = sigma
        self.method = method
        
        if self.sigma == 0:
            self.window_kernel = np.ones((self.window_size, self.window_size), dtype=np.float32) 
        else:
            self.create_gaussian_window()
            
        # First step --> Calculate gradient using sobel
        self.calculate_gradient(image)
        
        # Second step --> Get H matrix and calculate eigenvalues(lambda minus)
        self.calculate_H_matrix()
        
        # Third step --> Thresholding
        self.apply_thresholding()
        
        # Fourth step --> Non-maximum suppression
        self.apply_non_maximum_suppression()
        
        # convert to 8-bit representation for displaying
        self.corners = self.corners.astype(np.uint8)
        
        return self.corners
        
    def calculate_gradient(self, image):
        ''' Calculate the gradients of the image using Sobel '''
        
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.window_size)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.window_size)
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

                filtered_Ixx_window = Ixx_window * self.window_kernel
                filtered_Iyy_window = Iyy_window * self.window_kernel
                filtered_Ixy_window = Ixy_window * self.window_kernel
                
                Sxx = np.sum(filtered_Ixx_window) 
                Syy = np.sum(filtered_Iyy_window) 
                Sxy = np.sum(filtered_Ixy_window)
                
                H = np.array([[Sxx, Sxy],
                            [Sxy, Syy]])

                if self.method == 'Harris':
                    self.calculate_harris_response(H, y - half_window, x - half_window)
                elif self.method == 'lambda_minus':
                    self.calculate_eigenvalues(H, y - half_window, x - half_window)

        
    
    def calculate_eigenvalues(self, H_matrix, y, x):
        ''' Calculate the eigenvalues and eigenvectors of the Harris matrix '''
        
        eigenvalues = np.linalg.eigvals(H_matrix)
        self.lambda_minus[y, x] = min(eigenvalues)
    
    def calculate_harris_response(self, H_matrix, y, x):
        ''' Calculate the Harris response using the determinant and trace of the H matrix '''

        self.harris_response[y, x] = max((np.linalg.det(H_matrix) - 0.04 * (np.trace(H_matrix) ** 2)), 0)
        
      
    def apply_thresholding(self):
        ''' Apply thresholding to choose lambda minus '''
        
        if self.method == 'Lambda_minus':
           working_matrix = self.lambda_minus
            
        elif self.method == 'Harris':
            working_matrix = self.harris_response
            
        # Normalization of lambda minus for threshold (0:1)comparison
        working_matrix = (working_matrix - np.min(working_matrix)) / (np.max(working_matrix) - np.min(working_matrix)) 
        self.corners = np.where(working_matrix > self.threshold, 1, 0)
        
    def apply_non_maximum_suppression(self):
        ''' Apply non-maximum suppression to the corners to get the local maxima in each window '''
        if self.method == 'Harris':
            working_matrix = self.harris_response
        elif self.method == 'Lambda_minus':
            working_matrix = self.lambda_minus
            
        half_window = (self.window_size - 1) // 2
        padded_working_matrix = self.pad_image(working_matrix)
        for i in range(0, self.corners.shape[0]):
            for j in range(0, self.corners.shape[1]):
                if self.corners[i, j] == 1:
                    if working_matrix[i , j ] == np.max(padded_working_matrix[i:(i + self.window_size), j:(j + self.window_size)]):
                        self.corners[i, j] = 255
                    else:
                        self.corners[i, j] = 0
        
    def pad_image(self, image):
        ''' Pad the image to avoid border exceptions '''
        
        height, width = image.shape
        half_window = (self.window_size - 1) // 2
        padded_image = np.zeros(((height + 2 * half_window), (width + 2 * half_window)), dtype=image.dtype)
        padded_image[half_window:-half_window, half_window:-half_window] = image
        return padded_image    
    
    def create_gaussian_window(self):
        ''' Create a Gaussian window '''
        
        half_window = (self.window_size - 1) // 2
        x = np.linspace(-half_window, half_window, self.window_size)
        y = np.linspace(-half_window, half_window, self.window_size)
        X, Y = np.meshgrid(x, y)
    
        gaussian_window = np.exp(-(X**2 + Y**2) / (2 * self.sigma**2)) / (2 * np.pi * self.sigma**2)
        self.window_kernel = gaussian_window / np.sum(gaussian_window)    
    
    
    # def normalize_lambda_minus(self):
    #     ''' Normalize the lambda minus matrix between 0 and 255 '''
        
    #     self.lambda_minus = (self.lambda_minus - np.min(self.lambda_minus)) / (np.max(self.lambda_minus) - np.min(self.lambda_minus)) * 255
    #     self.corners = self.lambda_minus.astype(np.uint8)  # for 8-bit representation

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
    threshold = 0.1  # You can adjust this threshold
    window_size = 5  # You can adjust the window size
    sigma = 0
    features = feature_extractor.extraxt_lamda_minus(image,method="Harris", window_size=window_size, threshold=threshold, sigma=0)


    # Display the original image and the lambda_minus matrix
    cv2.imshow("Original Image", image)
    cv2.imshow("Lambda Minus Matrix", features)

    # Wait for a key press and close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()