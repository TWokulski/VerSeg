
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from GUI import MainMenuWidget
from GUI import TrainingWidget
from GUI import SegmentationWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setObjectName("MainWindow")
        self.resize(1024, 720)
        self.setMinimumSize(1024, 720)
        self.setMaximumSize(1024, 720)

        self.setWindowTitle("VerSeg")
        self.setWindowIcon(QIcon('icon.png'))
        self.central_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.choose_main_win()

    def choose_Training(self):
        create_model_widget = TrainingWidget(self)
        create_model_widget.back_to_menu_btn.clicked.connect(self.choose_main_win)
        self.central_widget.addWidget(create_model_widget)
        self.central_widget.setCurrentWidget(create_model_widget)

    def choose_Segmentation(self):
        segmentation_widget = SegmentationWidget(self)
        segmentation_widget.back_to_menu_btn.clicked.connect(self.choose_main_win)
        self.central_widget.addWidget(segmentation_widget)
        self.central_widget.setCurrentWidget(segmentation_widget)

    def choose_main_win(self):
        main_menu = MainMenuWidget(self)
        main_menu.training_button.clicked.connect(self.choose_Training)
        main_menu.segmentation_button.clicked.connect(self.choose_Segmentation)
        self.central_widget.addWidget(main_menu)
        self.central_widget.setCurrentWidget(main_menu)


if __name__ == '__main__':

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
