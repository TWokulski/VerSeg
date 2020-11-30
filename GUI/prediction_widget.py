from PyQt5 import Qt
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QTabWidget, QFileDialog, QTextEdit
import torch
import Mask_RCNN as algorithm
from PIL import Image, ImageQt
from PyQt5.QtCore import Qt
from torchvision import transforms
from PyQt5.QtCore import QThreadPool
import cv2
from GUI.ThreadClass import Worker


class PredictionWidget(QWidget):
    def __init__(self, parent=None):
        super(PredictionWidget, self).__init__(parent)
        self.model_path = ''
        self.image_to_predict = None
        self.setAcceptDrops(True)
        self.threadpool = QThreadPool()
        self.num_class = 0

        self.back_to_menu_btn = QPushButton("Back", self)
        self.back_to_menu_btn.setGeometry(20, 20, 100, 40)

        self.original_img_place = QLabel(self)
        self.original_img_place.setAlignment(Qt.AlignCenter)
        self.original_img_place.setText('\n\n Drop Image Here \n\n')
        self.original_img_place.setStyleSheet('''
                QLabel{
                    border: 4px dashed #aaa
                }
            ''')
        self.pred_img_place = QLabel(self)

        self.pictures_tabs = QTabWidget(self)
        self.pictures_tabs.setGeometry(127, 120, 750, 450)

        self.pictures_tabs.addTab(self.original_img_place, "Original image")
        self.pictures_tabs.addTab(self.pred_img_place, "Predicted mask")

        self.title = QLabel("VerSeg", self)
        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 40))
        self.title.setGeometry(200, 10, 400, 80)
        self.title.setStyleSheet("font-style: italic;\n"
                                 "font-weight: bold;")

        self.model_dir_lbl = QLabel("Directory for your model: ", self)
        self.model_dir_lbl.setWordWrap(True)
        self.model_dir_lbl.setFont(QFont('Arial', 10))
        self.model_dir_lbl.setStyleSheet("color: White")
        self.model_dir_lbl.setGeometry(127, 590, 200, 30)

        self.brows1_btn = QPushButton("Browse...", self)
        self.brows1_btn.clicked.connect(self.get_model_path)
        self.brows1_btn.setGeometry(337, 590, 80, 30)

        self.num_class_lbl = QLabel("Number of classes: ", self)
        self.num_class_lbl.setWordWrap(True)
        self.num_class_lbl.setFont(QFont('Arial', 10))
        self.num_class_lbl.setStyleSheet("color: White")
        self.num_class_lbl.setGeometry(127, 630, 200, 30)

        self.num_class_value = QTextEdit(self)
        self.num_class_value.setGeometry(337, 630, 80, 30)

        self.make_segm_btn = QPushButton("Segment", self)
        self.make_segm_btn.setGeometry(730, 590, 150, 40)
        self.make_segm_btn.setFont(QFont('Arial', 10))
        self.make_segm_btn.clicked.connect(self.process_prediction)

    def get_model_path(self):
        self.model_path, _ = QFileDialog.getOpenFileName(self, 'Choose a model file', '', 'Model files | *.pth;')

    def get_classes(self):
        try:
            self.num_class = int(self.num_class_value.toPlainText())
        except:
            self.num_class = 2

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.get_image_for_prediction(file_path)

        image = QPixmap(file_path)
        image = image.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
        self.original_img_place.setPixmap(image)

    def get_image_for_prediction(self, file_path):
        im = Image.open(file_path)
        im = im.convert("RGB")
        im = transforms.ToTensor()(im)
        self.image_to_predict = im

    def print_output(self, s):
        print(s)

    def training_complete(self):
        print("SEGMENTATION WAS PERFORMED")

    def process_prediction(self):
        worker = Worker(self.make_prediction)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.training_complete)
        self.threadpool.start(worker)

    def make_prediction(self):
        self.get_classes()
        if self.image_to_predict is not None and self.model_path != '':
            device = torch.device('cpu')
            model = algorithm.resnet50_for_mask_rcnn(True, self.num_class).to(device)

            user_model = torch.load(self.model_path, map_location=device)
            model.load_state_dict(user_model['model'])
            del user_model
            model.eval()

            with torch.no_grad():
                result = model(self.image_to_predict)
            result = {k: v.cpu() for k, v in result.items()}
            original_img, _, _, pred_box, pred_mask = algorithm.draw_image(self.image_to_predict, None, result, None)
            pred_image = cv2.addWeighted(original_img, 0.7, pred_mask, 0.7, 0)
            pred_image = cv2.addWeighted(pred_image, 0.7, pred_box, 1, 0)

            pred_image = ImageQt.ImageQt(Image.fromarray(pred_image.astype("uint8"), "RGB"))
            im = QPixmap.fromImage(pred_image).copy()
            im = im.scaledToWidth(self.pictures_tabs.width(), Qt.SmoothTransformation)
            self.pred_img_place.setPixmap(im)
        return "Done."

