from PyQt5 import QtGui, Qt
from PyQt5.QtGui import QIntValidator, QFont, QIcon, QPixmap
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QLineEdit, QComboBox, QTabWidget, QGridLayout, \
    QGroupBox, QFileDialog, QTextEdit, QApplication
import torch
import Mask_RCNN as algorithm
from PIL import Image, ImageQt
from PyQt5.QtCore import Qt
import cv2
import sys


class PredictionWidget(QWidget):
    def __init__(self, parent=None):
        super(PredictionWidget, self).__init__(parent)

        self.back_to_menu_btn = QPushButton("Back", self)
        self.back_to_menu_btn.setGeometry(20, 20, 100, 40)

        self.original_img_place = QLabel(self)
        self.pred_img_place = QLabel(self)

        self.pictures_tabs = QTabWidget(self)
        self.pictures_tabs.setGeometry(40, 120, 600, 400)

        self.pictures_tabs.addTab(self.original_img_place, "Original image")
        self.pictures_tabs.addTab(self.pred_img_place, "Predicted mask")

        self.title = QLabel("VerSeg", self)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 40))
        self.title.setGeometry(200, 10, 400, 80)
        self.title.setStyleSheet("font-style: italic;\n"
                                 "font-weight: bold;")
