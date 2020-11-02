import bisect
import glob
import re
import time
import torch
import Mask_RCNN as algorithm
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFileDialog, QWidget, QPushButton, QLabel, QComboBox, QTextEdit, QGroupBox, QGridLayout
import os


class TrainingWidget(QWidget):
    def __init__(self, parent=None):
        super(TrainingWidget, self).__init__(parent)

        self.best_model_by_maskAP = 0
        self.parameters = {}

        self.starting_lbl_y = 80
        self.starting_lbl_x = 20
        self.lbl_width = 300
        self.lbl_height = 30
        self.starting_input_x = 300

        self.dir_path = '/'

        self.back_to_menu_btn = QPushButton("Back", self)
        self.start_training_btn = QPushButton("START TRAINING", self)
        self.validating_btn = QPushButton("Validate parameters", self)
        self.validating_lbl = QLabel("Choose your parameters for training", self)
        self.data_params_box = QGroupBox("Your dataset", self)
        self.images_train_lbl = QLabel("Images in your training dataset: ", self)
        self.images_train_value_lbl = QLabel("82", self)
        self.images_val_lbl = QLabel("Images in your validation dataset: ", self)
        self.images_val_value_lbl = QLabel("9", self)
        self.device_lbl = QLabel("Your device: ", self)
        self.learning_rate_lbl = QLabel("Learning rate: ", self)
        self.learning_steps_lbl = QLabel("Learning steps: ", self)
        self.decay_lbl = QLabel("Decay: ", self)
        self.momentum_lbl = QLabel("Momentum: ", self)
        self.num_epoch_lbl = QLabel("Epochs: ", self)
        self.iterations_lbl = QLabel("Iterations per Epoch: ", self)
        self.dir_path_lbl = QLabel("Your dataset directory: ", self)
        self.class_num_lbl = QLabel("Number of classes in your dataset: ", self)
        self.seed_lbl = QLabel("Your seed for training: ", self)
        self.device_value = QComboBox(self)
        self.learning_rate_value = QTextEdit(self)
        self.learning_steps_value = QTextEdit(self)
        self.decay_value = QTextEdit(self)
        self.momentum_value = QTextEdit(self)
        self.epoch_value = QTextEdit(self)
        self.iterations_value = QTextEdit(self)
        self.brows_btn = QPushButton("Browse...", self)
        self.path_content = QTextEdit(self)
        self.class_value = QTextEdit(self)
        self.seed_value = QTextEdit(self)

        self.set_gui()

    def set_gui(self):
        self.set_config_section()
        self.set_header_section()
        self.set_params_section()

    def set_params_section(self):
        gbox = QGridLayout()
        self.data_params_box.setGeometry(674, 60, 280, 150)

        self.images_train_lbl.setWordWrap(True)
        self.images_train_lbl.setFont(QFont('Arial', 10))
        self.images_train_lbl.setStyleSheet("color: black")

        self.images_train_value_lbl.setWordWrap(True)
        self.images_train_value_lbl.setFont(QFont('Arial', 10))
        self.images_train_value_lbl.setStyleSheet("color: black")

        self.images_val_lbl.setWordWrap(True)
        self.images_val_lbl.setFont(QFont('Arial', 10))
        self.images_val_lbl.setStyleSheet("color: black")

        self.images_val_value_lbl.setWordWrap(True)
        self.images_val_value_lbl.setFont(QFont('Arial', 10))
        self.images_val_value_lbl.setStyleSheet("color: black")

        gbox.addWidget(self.images_train_lbl, 0, 0)
        gbox.addWidget(self.images_train_value_lbl, 0, 1)
        gbox.addWidget(self.images_val_lbl, 1, 0)
        gbox.addWidget(self.images_val_value_lbl, 1, 1)

        self.data_params_box.setLayout(gbox)

    def set_header_section(self):
        self.back_to_menu_btn.setGeometry(20, 20, 100, self.lbl_height + 10)

        self.start_training_btn.setGeometry(300, 650, 424, self.lbl_height + 20)
        self.start_training_btn.clicked.connect(self.train)

        self.validating_btn.setGeometry(self.starting_lbl_x + 20, self.starting_lbl_y + 420, 300, self.lbl_height + 10)

        self.validating_lbl.setWordWrap(True)
        self.validating_lbl.setFont(QFont('Arial', 20))
        self.validating_lbl.setStyleSheet("color: red")
        self.validating_lbl.setGeometry(275, 10, 624, self.lbl_height + 10)

    def set_config_section(self):
        self.device_lbl.setWordWrap(True)
        self.device_lbl.setFont(QFont('Arial', 10))
        self.device_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y, self.lbl_width, self.lbl_height)
        self.device_lbl.setStyleSheet("color: black")

        self.learning_rate_lbl.setWordWrap(True)
        self.learning_rate_lbl.setFont(QFont('Arial', 10))
        self.learning_rate_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 40, self.lbl_width,
                                           self.lbl_height)
        self.learning_rate_lbl.setStyleSheet("color: black")

        self.learning_steps_lbl.setWordWrap(True)
        self.learning_steps_lbl.setFont(QFont('Arial', 10))
        self.learning_steps_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 80, self.lbl_width,
                                            self.lbl_height)
        self.learning_steps_lbl.setStyleSheet("color: black")

        self.decay_lbl.setWordWrap(True)
        self.decay_lbl.setFont(QFont('Arial', 10))
        self.decay_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 120, self.lbl_width, self.lbl_height)
        self.decay_lbl.setStyleSheet("color: black")

        self.momentum_lbl.setWordWrap(True)
        self.momentum_lbl.setFont(QFont('Arial', 10))
        self.momentum_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 160, self.lbl_width, self.lbl_height)
        self.momentum_lbl.setStyleSheet("color: black")

        self.num_epoch_lbl.setWordWrap(True)
        self.num_epoch_lbl.setFont(QFont('Arial', 10))
        self.num_epoch_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 200, self.lbl_width, self.lbl_height)
        self.num_epoch_lbl.setStyleSheet("color: black")

        self.iterations_lbl.setWordWrap(True)
        self.iterations_lbl.setFont(QFont('Arial', 10))
        self.iterations_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 240, self.lbl_width, self.lbl_height)
        self.iterations_lbl.setStyleSheet("color: black")

        self.dir_path_lbl.setWordWrap(True)
        self.dir_path_lbl.setFont(QFont('Arial', 10))
        self.dir_path_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 280, self.lbl_width, self.lbl_height)
        self.dir_path_lbl.setStyleSheet("color: black")

        self.class_num_lbl.setWordWrap(True)
        self.class_num_lbl.setFont(QFont('Arial', 10))
        self.class_num_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 320, self.lbl_width, self.lbl_height)
        self.class_num_lbl.setStyleSheet("color: black")

        self.seed_lbl.setWordWrap(True)
        self.seed_lbl.setFont(QFont('Arial', 10))
        self.seed_lbl.setGeometry(self.starting_lbl_x, self.starting_lbl_y + 360, self.lbl_width, self.lbl_height)
        self.seed_lbl.setStyleSheet("color: black")

        self.device_value.addItems(["cuda", "cpu"])
        self.device_value.setGeometry(self.starting_input_x, self.starting_lbl_y, 80, self.lbl_height)
        self.learning_rate_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 40, 80, self.lbl_height)
        self.learning_steps_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 80, 80, self.lbl_height)
        self.decay_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 120, 80, self.lbl_height)
        self.momentum_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 160, 80, self.lbl_height)
        self.epoch_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 200, 80, self.lbl_height)
        self.iterations_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 240, 80, self.lbl_height)
        self.brows_btn.clicked.connect(self.get_path)
        self.brows_btn.setGeometry(235, self.starting_lbl_y + 280, 60, self.lbl_height)
        self.path_content.setGeometry(self.starting_input_x, self.starting_lbl_y + 280, 150, self.lbl_height)
        self.class_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 320, 80, self.lbl_height)
        self.seed_value.setGeometry(self.starting_input_x, self.starting_lbl_y + 360, 80, self.lbl_height)

    def get_path(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory", "C:\\")
        self.path_content.setText(self.dir_path)

    def get_seed(self):
        return int(self.seed_value.toPlainText())

    def get_epoch(self):
        return self.epoch_value

    def get_classes(self):
        return self.class_value

    def get_iterations(self):
        return self.iterations_value

    def get_momentum(self):
        return self.momentum_value

    def get_decay(self):
        return self.decay_value

    def get_learning_rate(self):
        return self.learning_rate_value

    def get_learning_steps(self):
        return self.learning_steps_value

    def get_device(self):
        return self.device_value

    def get_all_params(self):
        params = {
            "seed": self.get_seed(),
            "number_of_epochs": self.get_epoch(),
            "number_of_classes": self.get_classes(),
            "number_of_iterations": self.get_iterations(),
            "momentum": self.get_momentum(),
            "decay": self.get_decay(),
            "learning_rate": self.get_learning_rate(),
            "learning_steps": self.get_learning_steps(),
            "device": self.get_device(),
            "dataset_dir": self.dir_path,
            "publishing_losses_frequency": 100,
            "checkpoint_path": './ckpt',
            "learning_rate_lambda": 0.1,
            "model_path": './model',
            "iterations_to_warmup": 800,
            "result_path": './result'
        }
        return params

    def train(self):
        # self.parameters = self.get_all_params()
        self.parameters = {
            "seed": 1,
            "number_of_epochs": 2,
            "number_of_classes": 2,
            "number_of_iterations": 1,
            "momentum": 0.9,
            "decay": 0.0001,
            "learning_rate": 0.01,
            "learning_steps": [1, 5],
            "device": 'cpu',
            "dataset_dir": 'Dataset',
            "publishing_losses_frequency": 100,
            "checkpoint_path": './ckpt',
            "learning_rate_lambda": 0.1,
            "model_path": './model',
            "iterations_to_warmup": 800,
            "result_path": './result'
        }
        device = torch.device(self.parameters['device'])

        train_set = algorithm.COCODataset(self.parameters['dataset_dir'], "Train", train=True)
        indices = torch.randperm(len(train_set)).tolist()
        train_set = torch.utils.data.Subset(train_set, indices)

        val_set = algorithm.COCODataset(self.parameters['dataset_dir'], "Validation", train=True)
        model = algorithm.resnet50_for_mask_rcnn(True, self.parameters['number_of_classes']).to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=self.parameters['learning_rate'],
            momentum=self.parameters['momentum'],
            weight_decay=self.parameters['decay'])

        decrease = lambda x: self.parameters['learning_rate_lambda'] ** bisect.bisect(
            self.parameters['learning_steps'], x)

        starting_epoch = 0
        prefix, ext = os.path.splitext(self.parameters['checkpoint_path'])
        checkpoints = glob.glob(prefix + "-*" + ext)
        checkpoints.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        if checkpoints:
            checkpoint = torch.load(checkpoints[-1], map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            starting_epoch = checkpoint["epochs"]
            del checkpoint
            torch.cuda.empty_cache()

        since = time.time()
        print("\nalready trained: {} epochs; to {} epochs".format(starting_epoch, self.parameters['number_of_epochs']))

        for epoch in range(starting_epoch, self.parameters['number_of_epochs']):
            print("\nepoch: {}".format(epoch + 1))

            training_epoch_time = time.time()
            self.parameters['learning_epoch'] = decrease(epoch) * self.parameters['learning_rate']

            algorithm.train_epoch(model, optimizer, train_set, device, epoch, self.parameters)
            training_epoch_time = time.time() - training_epoch_time

            validation_epoch_time = time.time()
            eval_output = algorithm.evaluate(model, val_set, device, self.parameters)
            print(eval_output)
            validation_epoch_time = time.time() - validation_epoch_time

            trained_epoch = epoch + 1
            maskAP = eval_output.get_AP()
            print(maskAP)
            if maskAP['mask AP'] > self.best_model_by_maskAP:
                self.best_model_by_maskAP = maskAP['mask AP']
                algorithm.save_best(model, optimizer, trained_epoch,
                                    self.parameters['model_path'], eval_info=str(eval_output))

            algorithm.save_checkpoint(model, optimizer, trained_epoch,
                                self.parameters['checkpoint_path'], eval_info=str(eval_output))

            prefix, ext = os.path.splitext(self.parameters['checkpoint_path'])
            checkpoints = glob.glob(prefix + "-*" + ext)
            checkpoints.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
            n = 3
            if len(checkpoints) > n:
                for i in range(len(checkpoints) - n):
                    os.remove("{}".format(checkpoints[i]))

        total_training_time = time.time() - since
