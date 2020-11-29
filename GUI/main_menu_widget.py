import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QIntValidator, QFont, QIcon
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QWidget, QPushButton, QLabel, QLineEdit, QComboBox, \
    QMessageBox, QTextEdit

import subprocess
import os


class MainMenuWidget(QWidget):
    def __init__(self, parent=None):
        super(MainMenuWidget, self).__init__(parent)
        self.title = QLabel("VerSeg", self)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 60))
        self.title.setGeometry(40, 20, 400, 120)
        self.title.setStyleSheet("font-style: italic;\n"
                                 "font-weight: bold;")

        self.segmentation_button = QPushButton("Find vertebra", self)
        self.segmentation_button.setGeometry(172, 360, 320, 60)
        self.segmentation_button.setFont(QFont('Arial', 20))
        self.segmentation_button.setStyleSheet("background-color:white;\n"
                                               "color: black;\n"
                                               "font-weight: bold;"
                                               "")
        self.training_button = QPushButton("Create your model", self)
        self.training_button.setGeometry(532, 360, 320, 60)
        self.training_button.setFont(QFont('Arial', 20))
        self.training_button.setStyleSheet("background-color:white;\n"
                                           "color: black;\n"
                                           "font-weight: bold;"
                                           "")
        self.prediction_button = QPushButton("Test", self)
        self.prediction_button.setGeometry(532, 560, 320, 60)
        self.prediction_button.setFont(QFont('Arial', 20))
        self.prediction_button.setStyleSheet("background-color:white;\n"
                                           "color: black;\n"
                                           "font-weight: bold;"
                                           "")

