import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QStackedWidget, QPushButton
from PyQt5.QtWidgets import QApplication, QMainWindow , QPushButton, QStackedWidget , QFrame, QVBoxLayout, QComboBox , QLineEdit, QSlider
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc
from classes.controller import Controller
from classes.featureExtraxtion import FeatureExtraction
from classes.sift import SIFT
from classes.image import Image
compile_qrc()
from icons_setup.icons import *

from icons_setup.compiledIcons import *
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi('main.ui', self)
        self.setWindowTitle('Image Studio')
        self.setWindowIcon(QIcon('icons_setup\icons\logo.png'))

        # Add navigation Configuration
        self.modesCombobox = self.findChild(QComboBox, 'modesCombobox')
        self.modesCombobox.currentIndexChanged.connect(self.updateMode)

        self.modesStackedWidget = self.findChild(QStackedWidget, 'modesStackedWidget')
        self.framesStackedWidget = self.findChild(QStackedWidget, 'framesStackedWidget')
    
        # Initialize the Input Image Frame
        self.input_image_frame = self.findChild(QFrame , "extractFeaturesInputFrame")
                
        # Initialize Input Image Layout
        self.input_image_layout = QVBoxLayout(self.input_image_frame)
        
        self.input_image_layout = self.input_image_frame.layout()
        
        # Initialize Browse Image Button
        self.browse_image_button = self.findChild(QPushButton , "browse")
        self.browse_image_button.clicked.connect(self.browse_image)
        
        # Initialize Input Image
        self.input_image = Image()
        
        # Initialize Output Image
        self.output_image = Image()
        
        # Initialize Output Image Frame
        self.output_image_frame = self.findChild(QFrame , "extractFeaturesInputFrame")
        self.output_image_layout = self.output_image_frame.layout()
        self.output_image_frame.setLayout(self.output_image_layout)

        # linking for lambda minus
        self.lambda_minus_extraxtion = self.findChild(QPushButton, 'lambdaApply')    
        self.lambda_minus_extraxtion.clicked.connect(self.apply_lambda_minus_extraxtion)
        
        # Initialize lambda minus Parameters
        
        # # Threshold
        # self.lambda_threshold = 50
        # self.lambda_threshold_line_edit = self.findChild(QLineEdit , "snakeAlphaInput")
        # self.lambda_threshold_line_edit.setText("50")
        # self.lambda_threshold_line_edit.textChanged.connect(self.update_lambda_threshold)
       
        # # Window Size
        # self.lambda_window_size = 5
        # self.lambda_window_size_line_edit = self.findChild(QLineEdit , "snakeAlphaInput")
        # self.lambda_window_size_line_edit.setText("5")
        # self.lambda_window_size_line_edit.textChanged.connect(self.update_lambda_window_size)
      
        # initialize controller
        self.controller = Controller(self.input_image, self.output_image)

    def updateMode(self):
        current_mode = self.modesCombobox.currentText()
        if current_mode == 'Extract The Unique Features':
            self.framesStackedWidget.setCurrentIndex(0)
            self.modesStackedWidget.setCurrentIndex(0)
        elif current_mode == 'Generate Feature Descriptors':
            self.framesStackedWidget.setCurrentIndex(1)
            self.modesStackedWidget.setCurrentIndex(1)
        elif current_mode == 'Match Two Images':
            self.framesStackedWidget.setCurrentIndex(2)
            self.modesStackedWidget.setCurrentIndex(2)
        
    def browse_image(self):
        self.controller.browse_input_image()

    
 
    def apply_lambda_minus_extraxtion(self):
       self.controller.apply_lambda_minus_extraxtion(self.lambda_threshold, self.lambda_window_size)
    
    def update_lambda_threshold(self , text):
        try:
            self.lambda_threshold = int(text)
        except ValueError:
            self.lambda_threshold = 50
            self.lambda_threshold_line_edit.setText("50")
            
    def update_lambda_window_size(self , text):
        try:
            self.lambda_window_size = int(text)
        except ValueError:
            self.lambda_window_size = 5
            self.lambda_window_size_line_edit.setText("5")
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   