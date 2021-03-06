from PyQt5 import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QTabWidget, QGridLayout, \
    QGroupBox, QFileDialog, QTextEdit, QApplication
import torch
import Mask_RCNN as algorithm
from PIL import Image, ImageQt
from PyQt5.QtCore import Qt, QThreadPool
from GUI.ThreadClass import Worker
import cv2
import sys


class SegmentationWidget(QWidget):
    def __init__(self, parent=None):
        super(SegmentationWidget, self).__init__(parent)
        self.threadpool = QThreadPool()

        self.org_im_list = []
        self.gt_list = []
        self.pred_list = []
        self.ap_scores = []
        self.f1_scores = []
        self.present_index = 0

        self.model_path = ''
        self.dir_path = ''

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
        self.eval_box.setGeometry(674, 135, 250, 160)

        self.ap_mask_text_lbl = QLabel("Mask AP: ", self)
        self.ap_mask_text_lbl.setWordWrap(True)
        self.ap_mask_text_lbl.setFont(QFont('Arial', 10))
        self.ap_mask_text_lbl.setStyleSheet("color: Gray")

        self.ap_mask_value_lbl = QLabel("", self)
        self.ap_mask_value_lbl.setWordWrap(True)
        self.ap_mask_value_lbl.setFont(QFont('Arial', 10))
        self.ap_mask_value_lbl.setStyleSheet("color: Gray")

        self.ap_box_text_lbl = QLabel("Box AP: ", self)
        self.ap_box_text_lbl.setWordWrap(True)
        self.ap_box_text_lbl.setFont(QFont('Arial', 10))
        self.ap_box_text_lbl.setStyleSheet("color: Gray")

        self.ap_box_value_lbl = QLabel("", self)
        self.ap_box_value_lbl.setWordWrap(True)
        self.ap_box_value_lbl.setFont(QFont('Arial', 10))
        self.ap_box_value_lbl.setStyleSheet("color: Gray")

        self.f1_box_text_lbl = QLabel("Box F Score: ", self)
        self.f1_box_text_lbl.setWordWrap(True)
        self.f1_box_text_lbl.setFont(QFont('Arial', 10))
        self.f1_box_text_lbl.setStyleSheet("color: Gray")

        self.f1_box_value_lbl = QLabel("", self)
        self.f1_box_value_lbl.setWordWrap(True)
        self.f1_box_value_lbl.setFont(QFont('Arial', 10))
        self.f1_box_value_lbl.setStyleSheet("color: Gray")

        self.f1_mask_text_lbl = QLabel("Mask F Score: ", self)
        self.f1_mask_text_lbl.setWordWrap(True)
        self.f1_mask_text_lbl.setFont(QFont('Arial', 10))
        self.f1_mask_text_lbl.setStyleSheet("color: Gray")

        self.f1_mask_value_lbl = QLabel("", self)
        self.f1_mask_value_lbl.setWordWrap(True)
        self.f1_mask_value_lbl.setFont(QFont('Arial', 10))
        self.f1_mask_value_lbl.setStyleSheet("color: Gray")

        gbox.addWidget(self.ap_mask_text_lbl, 0, 0)
        gbox.addWidget(self.ap_mask_value_lbl, 0, 1)
        gbox.addWidget(self.ap_box_text_lbl, 1, 0)
        gbox.addWidget(self.ap_box_value_lbl, 1, 1)
        gbox.addWidget(self.f1_mask_text_lbl, 2, 0)
        gbox.addWidget(self.f1_mask_value_lbl, 2, 1)
        gbox.addWidget(self.f1_box_text_lbl, 3, 0)
        gbox.addWidget(self.f1_box_value_lbl, 3, 1)

        self.eval_box.setLayout(gbox)

        self.previous_img_btn = QPushButton("Previous image", self)
        self.previous_img_btn.setGeometry(85, 540, 250, 30)
        self.previous_img_btn.setFont(QFont('Arial', 10))

        self.next_img_btn = QPushButton("Next image", self)
        self.next_img_btn.setGeometry(345, 540, 250, 30)
        self.next_img_btn.setFont(QFont('Arial', 10))

        self.next_img_btn.clicked.connect(self.get_next_image)
        self.previous_img_btn.clicked.connect(self.get_previous_image)

        self.make_segm_btn = QPushButton("Segment", self)
        self.make_segm_btn.setGeometry(674, 530, 250, 40)
        self.make_segm_btn.setFont(QFont('Arial', 10))

        self.make_segm_btn.clicked.connect(self.process_visualise)
        self.make_segm_btn.setDisabled(False)

        self.param_box = QGroupBox("Dataset info", self)
        gbox2 = QGridLayout()
        self.param_box.setGeometry(674, 300, 250, 180)

        self.data_dir_lbl = QLabel("Directory for your dataset: ", self)
        self.data_dir_lbl.setWordWrap(True)
        self.data_dir_lbl.setFont(QFont('Arial', 10))
        self.data_dir_lbl.setStyleSheet("color: Gray")

        self.model_dir_lbl = QLabel("Directory for your model", self)
        self.model_dir_lbl.setWordWrap(True)
        self.model_dir_lbl.setFont(QFont('Arial', 10))
        self.model_dir_lbl.setStyleSheet("color: Gray")

        self.val_samples_lbl = QLabel("Number of smaples: ", self)
        self.val_samples_lbl.setWordWrap(True)
        self.val_samples_lbl.setFont(QFont('Arial', 10))
        self.val_samples_lbl.setStyleSheet("color: Gray")

        self.brows1_btn = QPushButton("Browse...", self)
        self.brows1_btn.clicked.connect(self.get_dir_path)
        self.brows2_btn = QPushButton("Browse...", self)
        self.brows2_btn.clicked.connect(self.get_model_path)

        self.val_samples_value = QTextEdit(self)

        gbox2.setColumnMinimumWidth(0, 240)
        gbox2.addWidget(self.data_dir_lbl, 0, 0)
        gbox2.addWidget(self.brows1_btn, 0, 1)
        gbox2.addWidget(self.model_dir_lbl, 1, 0)
        gbox2.addWidget(self.brows2_btn, 1, 1)
        gbox2.addWidget(self.val_samples_lbl, 2, 0)
        gbox2.addWidget(self.val_samples_value, 2, 1)

        self.param_box.setLayout(gbox2)

    def get_dir_path(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory", "./")
        if self.dir_path and self.model_path != '':
            self.make_segm_btn.setDisabled(False)

    def get_model_path(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, 'Choose a model file', '', 'Model files | *.pth;')
        if self.dir_path and self.model_path != '':
            self.make_segm_btn.setDisabled(False)

    def get_val_samples(self):
        try:
            samples = self.val_samples_value.toPlainText()
            if samples == '':
                samples = 1
            else:
                samples = int(samples)
        except:
           return 1
        return samples

    def get_next_image(self):
        self.previous_img_btn.setDisabled(False)
        if self.present_index < (len(self.org_im_list) - 1):
            self.present_index += 1
            self.update_img_place(self.present_index)
            if self.present_index == len(self.org_im_list) - 1:
                self.next_img_btn.setDisabled(True)

    def get_previous_image(self):
        self.next_img_btn.setDisabled(False)
        if self.present_index > 0:
            self.present_index -= 1
            self.update_img_place(self.present_index)
            if self.present_index == 0:
                self.previous_img_btn.setDisabled(True)

    def print_output(self, s):
        print(s)

    def training_complete(self):
        print("SEGMENTATION WAS PERFORMED")

    def process_visualise(self):
        worker = Worker(self.visualise)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.training_complete)
        self.threadpool.start(worker)

    def visualise(self):

        self.org_im_list = []
        self.gt_list = []
        self.pred_list = []

        try:
            val_samples = self.get_val_samples()
            data_dir = self.dir_path
            model_path = self.model_path

            device = torch.device('cpu')

            dataset = algorithm.COCODataset(data_dir, 'Validation', True)
            classes = dataset.classes
            coco = dataset.coco
            iou_types = ['bbox', 'segm']
            evaluator = algorithm.CocoEvaluator(coco, iou_types)

            ann_labels = dataset.annotations_labels
            coco_results = []

            indices = torch.randperm(len(dataset)).tolist()
            dataset = torch.utils.data.Subset(dataset, indices[:val_samples])
            model = algorithm.resnet50_for_mask_rcnn(True, len(coco.cats)).to(device)

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
                coco_results.extend(algorithm.prepare_for_coco(res, ann_labels))
                output = evaluate(evaluator, coco_results)
                self.ap_scores.append(output.get_AP())
                self.f1_scores.append(output.get_AF())

                original_img, gt_box, gt_mask, pred_box, pred_mask = algorithm.draw_image(image, target, result, classes)
                gt_image = cv2.addWeighted(original_img, 0.7, gt_mask, 0.7, 0)
                gt_image = cv2.addWeighted(gt_image, 0.7, gt_box, 1, 0)
                pred_image = cv2.addWeighted(original_img, 0.7, pred_mask, 0.7, 0)
                pred_image = cv2.addWeighted(pred_image, 0.7, pred_box, 1, 0)

                original_img = ImageQt.ImageQt(Image.fromarray(original_img.astype("uint8"), "RGB"))
                gt_image = ImageQt.ImageQt(Image.fromarray(gt_image.astype("uint8"), "RGB"))
                pred_image = ImageQt.ImageQt(Image.fromarray(pred_image.astype("uint8"), "RGB"))

                self.org_im_list.append(original_img)
                self.gt_list.append(gt_image)
                self.pred_list.append(pred_image)
            self.update_img_place(0)
            self.previous_img_btn.setDisabled(True)
        except:
            return 'Failed.'
        return 'Done.'

    def update_img_place(self, index):
        im1 = QPixmap.fromImage(self.org_im_list[index]).copy()
        im2 = QPixmap.fromImage(self.pred_list[index]).copy()
        im3 = QPixmap.fromImage(self.gt_list[index]).copy()
        im1 = im1.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
        im2 = im2.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
        im3 = im3.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
        self.original_img_place.setPixmap(im1)
        self.pred_img_place.setPixmap(im2)
        self.gt_img_place.setPixmap(im3)

        ap = self.ap_scores[index]
        f1 = self.f1_scores[index]

        self.ap_box_value_lbl.setText('{}'.format(ap["bbox AP"]))
        self.ap_mask_value_lbl.setText('{}'.format(ap["mask AP"]))
        self.f1_box_value_lbl.setText('{}'.format(f1['bbox FScore']))
        self.f1_mask_value_lbl.setText('{}'.format(f1['mask FScore']))
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


def evaluate(evaluator, results):
    coco_evaluator = evaluator
    coco_evaluator.accumulate_results(results)

    temp = sys.stdout
    sys.stdout = algorithm.CocoConversion()
    coco_evaluator.summarize()
    output = sys.stdout
    sys.stdout = temp

    return output



