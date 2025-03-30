import  numpy as np
import cv2
class FeatureExtraction:
    def __init__(self):
        ''' Harris and lambda minus parameters'''
        self.window_size = None
        self.features = None
        self.threshold = None
        self.H_matrix = None
        self.lambda_minus = None
        
    def extraxt_lamda_minus(self, image, threshold = 50, window_size = 5):    
        ''' Extract lambda minus features '''
        
        # Initalization
        self.lambda_minus = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        self.window_size = window_size
        self.threshold = threshold
        
        # First step --> Apply Gaussian smoothing
        blurred_image = self.blur_image(image)
        
        # Second step --> Calculate gradient using sobel
        self.calculate_gradient(blurred_image)
        
        # Third step --> Get Harris matrix and calculate eigenvalues(lambda minus)
        self.calculate_H_matrix(blurred_image)
        
        # Fourth step --> Thresholding
        self.apply_thresholding()     
           
        return self.features
        
    
    def blur_image(self, image):
        ''' Apply smoothing using Gaussian  '''
        
        blurred_image = cv2.GaussianBlur(image, (self.window_size, self.window_size), 1)
        return blurred_image
    
    def calculate_gradient(self, image):
        ''' Calculate the gradients of the image using Sobel '''
        
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        self.Ixx = sobel_x ** 2
        self.Iyy = sobel_y ** 2
        self.Ixy = sobel_x * sobel_y      
        self.Iyx = sobel_y * sobel_x
        
        
    def calculate_H_matrix(self, image):
        ''' Calculate the Harris matrix '''
        
        height, width = image.shape
        half_window = (self.window_size - 1) // 2  

        for x in range(half_window, height - half_window):
            for y in range(half_window, width - half_window):
                
                Ixx_window = self.Ixx[x - half_window : x + half_window + 1, y - half_window : y + half_window + 1]
                Iyy_window = self.Iyy[x - half_window : x + half_window + 1, y - half_window : y + half_window + 1]
                Ixy_window = self.Ixy[x - half_window : x + half_window + 1, y - half_window : y + half_window + 1]
                Iyx_window = self.Iyx[x - half_window : x + half_window + 1, y - half_window : y + half_window + 1]

                Sxx = np.sum(Ixx_window)
                Syy = np.sum(Iyy_window)
                Sxy = np.sum(Ixy_window)
                Syx = np.sum(Iyx_window)
                
                H = np.array([[Sxx, Sxy],
                            [Syx, Syy]])

                self.calculate_eigenvalues(H, y, x)

        
    
    def calculate_eigenvalues(self, H_matrix, y, x):
        ''' Calculate the eigenvalues and eigenvectors of the Harris matrix '''
        
        eigenvalues = np.linalg.eigvals(H_matrix)
    
        self.lambda_minus[y, x] = min(eigenvalues)

    
    def apply_thresholding(self):
        ''' Apply thresholding to choose lambda minus '''
        
        self.features = np.where(self.lambda_minus > self.threshold, 255, 0)