import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QComboBox, QStackedWidget
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon
from helper_functions.compile_qrc import compile_qrc

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
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())   