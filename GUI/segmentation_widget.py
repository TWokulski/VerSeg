import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QIntValidator, QFont, QIcon
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QLineEdit, QComboBox, QTabWidget, QGraphicsView, QGridLayout,\
    QGroupBox, QFileDialog, QTextEdit

import subprocess
import os


class SegmentationWidget(QWidget):
    def __init__(self, parent=None):
        super(SegmentationWidget, self).__init__(parent)

        self.back_to_menu_btn = QPushButton("Back", self)
        self.back_to_menu_btn.setGeometry(20, 20, 100, 40)

        self.original_img_place = QGraphicsView(self)
        self.mask_img_place = QGraphicsView(self)

        self.pictures_tabs = QTabWidget(self)
        self.pictures_tabs.setGeometry(40, 120, 600, 450)

        self.pictures_tabs.addTab(self.original_img_place, "Original image")
        self.pictures_tabs.addTab(self.mask_img_place, "Predicted mask")

        self.eval_box = QGroupBox("Evaluation Metrics", self)
        gbox = QGridLayout()
        self.eval_box.setGeometry(674, 135, 280, 150)

        self.ap_text_lbl = QLabel("AP for your image: ", self)
        self.ap_text_lbl.setWordWrap(True)
        self.ap_text_lbl.setFont(QFont('Arial', 10))
        self.ap_text_lbl.setStyleSheet("color: black")

        self.ap_value_lbl = QLabel("82", self)
        self.ap_value_lbl.setWordWrap(True)
        self.ap_value_lbl.setFont(QFont('Arial', 10))
        self.ap_value_lbl.setStyleSheet("color: black")

        self.iou_text_lbl = QLabel("IoU for your image: ", self)
        self.iou_text_lbl.setWordWrap(True)
        self.iou_text_lbl.setFont(QFont('Arial', 10))
        self.iou_text_lbl.setStyleSheet("color: black")

        self.iou_value_lbl = QLabel("9", self)
        self.iou_value_lbl.setWordWrap(True)
        self.iou_value_lbl.setFont(QFont('Arial', 10))
        self.iou_value_lbl.setStyleSheet("color: black")

        gbox.addWidget(self.ap_text_lbl, 0, 0)
        gbox.addWidget(self.ap_value_lbl, 0, 1)
        gbox.addWidget(self.iou_text_lbl, 1, 0)
        gbox.addWidget(self.iou_value_lbl, 1, 1)

        self.eval_box.setLayout(gbox)

        self.next_img_btn = QPushButton("Previous image", self)
        self.next_img_btn.setGeometry(85, 580, 250, 30)
        self.next_img_btn.setFont(QFont('Arial', 10))

        self.next_img_btn = QPushButton("Next image", self)
        self.next_img_btn.setGeometry(345, 580, 250, 30)
        self.next_img_btn.setFont(QFont('Arial', 10))

        self.make_segm_btn = QPushButton("Segment", self)
        self.make_segm_btn.setGeometry(674, 530, 280, 40)
        self.make_segm_btn.setFont(QFont('Arial', 10))

        self.param_box = QGroupBox("Evaluation Metrics", self)
        gbox2 = QGridLayout()
        self.param_box.setGeometry(674, 300, 280, 150)

        self.data_dir_lbl = QLabel("Directory for your dataset: ", self)
        self.data_dir_lbl.setWordWrap(True)
        self.data_dir_lbl.setFont(QFont('Arial', 10))
        self.data_dir_lbl.setStyleSheet("color: black")

        self.model_dir_lbl = QLabel("Directory for your model", self)
        self.model_dir_lbl.setWordWrap(True)
        self.model_dir_lbl.setFont(QFont('Arial', 10))
        self.model_dir_lbl.setStyleSheet("color: black")

        self.num_class_lbl = QLabel("Number of classes: ", self)
        self.num_class_lbl.setWordWrap(True)
        self.num_class_lbl.setFont(QFont('Arial', 10))
        self.num_class_lbl.setStyleSheet("color: black")

        self.brows1_btn = QPushButton("Browse...", self)
        self.brows1_btn.clicked.connect(self.get_dir_path)
        self.brows2_btn = QPushButton("Browse...", self)
        self.brows2_btn.clicked.connect(self.get_model_path)

        self.num_class_value = QTextEdit(self)

        gbox2.setColumnMinimumWidth(0, 240)
        gbox2.addWidget(self.data_dir_lbl, 0, 0)
        gbox2.addWidget(self.brows1_btn, 0, 1)
        gbox2.addWidget(self.model_dir_lbl, 1, 0)
        gbox2.addWidget(self.brows2_btn, 1, 1)
        gbox2.addWidget(self.num_class_lbl, 2, 0)
        gbox2.addWidget(self.num_class_value, 2, 1)

        self.param_box.setLayout(gbox2)

    def get_dir_path(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory", "./")
        self.path_content.setText(self.dir_path)

    def get_model_path(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, 'Choose a model file', '', 'Model files | *.pth;')
        self.path_content.setText(self.model_path)