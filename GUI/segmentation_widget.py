from PyQt5 import QtGui, Qt
from PyQt5.QtGui import QIntValidator, QFont, QIcon, QPixmap
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QLineEdit, QComboBox, QTabWidget, QGridLayout, \
    QGroupBox, QFileDialog, QTextEdit, QApplication
import torch
import Mask_RCNN as algorithm
from PIL import Image, ImageQt
from PyQt5.QtCore import Qt
import cv2


class SegmentationWidget(QWidget):
    def __init__(self, parent=None):
        super(SegmentationWidget, self).__init__(parent)

        self.back_to_menu_btn = QPushButton("Back", self)
        self.back_to_menu_btn.setGeometry(20, 20, 100, 40)

        self.original_img_place = QLabel(self)
        self.pred_img_place = QLabel(self)
        self.gt_img_place = QLabel(self)

        self.pictures_tabs = QTabWidget(self)
        self.pictures_tabs.setGeometry(40, 120, 600, 400)

        self.pictures_tabs.addTab(self.original_img_place, "Original image")
        self.pictures_tabs.addTab(self.pred_img_place, "Predicted mask")
        self.pictures_tabs.addTab(self.gt_img_place, "Target mask")

        self.title = QLabel("VerSeg", self)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 40))
        self.title.setGeometry(200, 10, 400, 80)
        self.title.setStyleSheet("font-style: italic;\n"
                                 "font-weight: bold;")

        self.eval_box = QGroupBox("Evaluation Metrics", self)
        gbox = QGridLayout()
        self.eval_box.setGeometry(674, 135, 280, 120)

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

        self.make_segm_btn.clicked.connect(self.visualise)

        self.param_box = QGroupBox("Dataset info", self)
        gbox2 = QGridLayout()
        self.param_box.setGeometry(674, 300, 280, 180)

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

    def get_num_classes(self):
        return int(self.num_class_value.toPlainText())

    def visualise(self):

        org_im_list = []
        gt_list = []
        pred_list = []

        val_samples = 1
        data_dir = 'Dataset'
        num_classes = 2
        model_path = 'ckpt/bestbest-98.pth'

        device = torch.device('cpu')

        dataset = algorithm.COCODataset(data_dir, 'Validation', True)
        classes = dataset.classes
        coco = dataset.coco
        iou_types = ['bbox', 'segm']
        evaluator = algorithm.CocoEvaluator(coco, iou_types)

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:val_samples])
        model = algorithm.resnet50_for_mask_rcnn(True, num_classes).to(device)

        user_model = torch.load(model_path, map_location=device)
        model.load_state_dict(user_model['model'])
        del user_model
        model.eval()

        for (image, target) in dataset:
            with torch.no_grad():
                result = model(image)
            result = {k: v.cpu() for k, v in result.items()}
            res = {target['image_id'].item(): result}
            evaluator.update(res)
            original_img, gt_box, gt_mask, pred_box, pred_mask = algorithm.draw_image(image, target, result, classes)
            gt_image = cv2.addWeighted(original_img, 0.7, gt_mask, 0.7, 0)
            gt_image = cv2.addWeighted(gt_image, 0.7, gt_box, 1, 0)
            pred_image = cv2.addWeighted(original_img, 0.7, pred_mask, 0.7, 0)
            pred_image = cv2.addWeighted(pred_image, 0.7, pred_box, 1, 0)

            original_img = ImageQt.ImageQt(Image.fromarray(original_img.astype("uint8"), "RGB"))
            gt_image = ImageQt.ImageQt(Image.fromarray(gt_image.astype("uint8"), "RGB"))
            pred_image = ImageQt.ImageQt(Image.fromarray(pred_image.astype("uint8"), "RGB"))

            org_im_list.append(original_img)
            gt_list.append(gt_image)
            pred_list.append(pred_image)

        im1 = QPixmap.fromImage(org_im_list[0]).copy()
        im2 = QPixmap.fromImage(pred_list[0]).copy()
        im3 = QPixmap.fromImage(gt_list[0]).copy()
        im1 = im1.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
        im2 = im2.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
        im3 = im3.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
        self.original_img_place.setPixmap(im1)
        self.pred_img_place.setPixmap(im2)
        self.gt_img_place.setPixmap(im3)
        QApplication.processEvents()

    def show_images(self, image, target):
        with torch.no_grad():
            result = self.model(image)

        result = {k: v.cpu() for k, v in result.items()}
        res = {target['image_id'].item(): result}
        self.evaluator.update(res)

        gt_true = algorithm.show_single_target(target['masks'], image)
        prediction = algorithm.show_single(image, result, self.classes)
        return gt_true, prediction

