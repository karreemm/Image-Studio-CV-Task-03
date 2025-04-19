import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QStackedWidget, QPushButton
from PyQt5.QtWidgets import QApplication, QMainWindow , QPushButton, QStackedWidget , QFrame, QVBoxLayout, QComboBox , QLineEdit, QLabel
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc
from classes.controller import Controller
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
    
        # Extaction Mode
        self.extractFeaturesInputFrame = self.findChild(QFrame, 'extractFeaturesInputFrame')
        self.extractFeaturesOutputFrame = self.findChild(QFrame, 'extractFeaturesOutputFrame')

        # SIFT Mode
        self.siftInputFrame = self.findChild(QFrame, 'siftInputFrame')
        self.siftOutputFrame = self.findChild(QFrame, 'siftOutputFrame')

        # Matching Mode
        self.matchingImage1Frame = self.findChild(QFrame, 'matchingImage1Frame')
        self.matchingImage2Frame = self.findChild(QFrame, 'matchingImage2Frame')
        self.matchingResultFrame = self.findChild(QFrame, 'matchingResultFrame')

        # Array of frames
        self.frames = [self.extractFeaturesInputFrame, self.extractFeaturesOutputFrame, self.siftInputFrame, self.siftOutputFrame, self.matchingImage1Frame, self.matchingImage2Frame, self.matchingResultFrame]
        self.frame_labels = []  
        
        # Apply layout and label setup for each frame
        for frame in self.frames:

            label = QLabel(frame)
            layout = QVBoxLayout(frame)
            layout.addWidget(label)
            frame.setLayout(layout)
            self.frame_labels.append(label)
            label.setScaledContents(True)

        # Initialize Controller
        self.controller = Controller(self.frame_labels)

        # Initialize Browse Image Button
        self.browse_image_button = self.findChild(QPushButton , "browse")
        self.browse_image_button.clicked.connect(self.browse_image)

        # Initialize Apply Harris Extraction Button\
        self.harrisApply = self.findChild(QPushButton , "harrisApply")
        self.harrisApply.clicked.connect(self.apply_harris_extraxtion)

        # Initialize Apply Lambda -ve Extraction Button
        self.lambdaApply = self.findChild(QPushButton , "lambdaApply")
        self.lambdaApply.clicked.connect(self.apply_lambda_minus_extraxtion)

        self.thresholdInputLambda = self.findChild(QLineEdit , "thresholdInputLambda")
        self.thresholdInputLambda.setText("0.1")
        self.lambda_threshold = float(self.thresholdInputLambda.text())
        self.thresholdInputLambda.textChanged.connect(self.update_lambda_minus_parameters)

        self.sigmaInputLambda = self.findChild(QLineEdit , "sigmaInputLambda")
        self.sigmaInputLambda.setText("0.5")
        self.sigma = float(self.sigmaInputLambda.text())
        self.sigmaInputLambda.textChanged.connect(self.update_lambda_minus_parameters)

        self.windowSizeInputLambda = self.findChild(QLineEdit , "windowSizeInputLambda")
        self.windowSizeInputLambda.setText("3")
        self.lambda_window_size = int(self.windowSizeInputLambda.text())
        self.windowSizeInputLambda.textChanged.connect(self.update_lambda_minus_parameters)

        # Initialize Apply SIFT Extraction Button
        self.siftApply = self.findChild(QPushButton , "siftApply")
        self.siftApply.clicked.connect(self.apply_sift)

        self.sigmaInputSift = self.findChild(QLineEdit , "sigmaInputSift")
        self.sigmaInputSift.setText("1.6")
        self.sigmaSift = float(self.sigmaInputSift.text())
        self.sigmaInputSift.textChanged.connect(self.update_sift_parameters)

        self.numberOfOcatvesInputSift = self.findChild(QLineEdit , "numberOfOcatvesInputSift")
        self.numberOfOcatvesInputSift.setText("4")
        self.numOfOctavesSift = int(self.numberOfOcatvesInputSift.text())
        self.numberOfOcatvesInputSift.textChanged.connect(self.update_sift_parameters)

        # Marching Image 1 Button
        self.matchingBrowseImage1 = self.findChild(QPushButton , "matchingBrowseImage1")
        self.matchingBrowseImage1.clicked.connect(self.controller.browse_matching_image1)

        # Marching Image 2 Button
        self.matchingBrowseImage2 = self.findChild(QPushButton , "matchingBrowseImage2")
        self.matchingBrowseImage2.clicked.connect(self.controller.browse_matching_image2)

        # Initialize Matching Parameters
        self.thresholdInputMatching = self.findChild(QLineEdit , "thresholdInputMatching")
        self.thresholdInputMatching.setText("0.8")
        self.matching_threshold = float(self.thresholdInputMatching.text())
        self.thresholdInputMatching.textChanged.connect(self.update_matching_parameters)

        self.numOfMatchesInputMatching = self.findChild(QLineEdit , "numOfMatchesInputMatching")
        self.numOfMatchesInputMatching.setText("150")
        self.numOfMatches = int(self.numOfMatchesInputMatching.text())
        self.numOfMatchesInputMatching.textChanged.connect(self.update_matching_parameters)

        # Initialize Apply SSD Matching Button
        self.ssdApply = self.findChild(QPushButton , "ssdApply")
        self.ssdApply.clicked.connect(self.apply_ssd_matching)

        # Initialize Apply NCC Matching Button
        self.nccApply = self.findChild(QPushButton , "correlationsApply")
        self.nccApply.clicked.connect(self.apply_ncc_matching)

    
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

        
    def apply_harris_extraxtion(self):
        pass

    def apply_lambda_minus_extraxtion(self):
        print("parameters: ", self.lambda_window_size, self.lambda_threshold, self.sigma)
        self.controller.apply_lambda_minus_extraction(self.lambda_window_size, self.lambda_threshold, self.sigma)

    def update_lambda_minus_parameters(self):
        if self.windowSizeInputLambda.text() == "" or self.windowSizeInputLambda.text().isalpha():
            self.windowSizeInputLambda.setText("3")
        if self.thresholdInputLambda.text() == "" or self.thresholdInputLambda.text().isalpha():
            self.thresholdInputLambda.setText("0.1")
        if self.sigmaInputLambda.text() == "" or self.sigmaInputLambda.text().isalpha():
            self.sigmaInputLambda.setText("0.5")
        self.lambda_window_size = int(self.windowSizeInputLambda.text())
        self.lambda_threshold = float(self.thresholdInputLambda.text())
        self.sigma = float(self.sigmaInputLambda.text())

    def apply_sift(self):
        self.controller.apply_sift(self.sigmaSift, self.numOfOctavesSift)

    def update_sift_parameters(self):
        if self.numberOfOcatvesInputSift.text() == "" or self.numberOfOcatvesInputSift.text().isalpha():
            self.numberOfOcatvesInputSift.setText("4")
        if self.sigmaInputSift.text() == "" or self.sigmaInputSift.text().isalpha():
            self.sigmaInputSift.setText("1.6")
        self.numOfOctavesSift = int(self.numberOfOcatvesInputSift.text())
        self.sigmaSift = float(self.sigmaInputSift.text())

    def apply_ssd_matching(self):
        self.controller.match_images(method='ssd', threshold=self.matching_threshold, num_matches=self.numOfMatches)

    def apply_ncc_matching(self):
        self.controller.match_images(method='ncc')

    def update_matching_parameters(self):
        if self.numOfMatchesInputMatching.text() == "" or self.numOfMatchesInputMatching.text().isalpha():
            self.numOfMatchesInputMatching.setText("150")
        if self.thresholdInputMatching.text() == "" or self.thresholdInputMatching.text().isalpha():
            self.thresholdInputMatching.setText("0.8")
        self.numOfMatches = int(self.numOfMatchesInputMatching.text())
        self.matching_threshold = float(self.thresholdInputMatching.text())

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   