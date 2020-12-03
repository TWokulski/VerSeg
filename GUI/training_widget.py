import os
import bisect
import glob
import re
import time
import torch
import Mask_RCNN as algorithm
from Config import Configuration
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFileDialog, QWidget, QPushButton, QLabel, QComboBox, QTextEdit, QGroupBox, QGridLayout, \
    QApplication, QProgressBar
from .ThreadClass import *
from PyQt5.QtCore import QThreadPool


class TrainingWidget(QWidget):
    def __init__(self, parent=None):
        super(TrainingWidget, self).__init__(parent)

        self.best_model_by_maskF = 0
        self.parameters = {}

        self.threadpool = QThreadPool()

        self.starting_lbl_y = 80
        self.starting_lbl_x = 20
        self.lbl_width = 300
        self.lbl_height = 30
        self.starting_input_x = 300
        self.input_width = 100
        self.dir_path = '/'
        self.present_epoch = 0

        self.title = QLabel("VerSeg", self)
        self.back_to_menu_btn = QPushButton("Back", self)

        self.start_training_btn = QPushButton("START TRAINING", self)
        self.validating_btn = QPushButton("Validate parameters", self)

        self.use_default_params_btn = QPushButton("Use default parameters", self)
        self.clear_btn = QPushButton("Clear configuration", self)
        self.start_fresh_box = QComboBox(self)

        self.training_params_box = QGroupBox("Your training parameters", self)
        self.warning_lbl = QLabel("", self)

        self.data_params_box = QGroupBox("Your dataset", self)
        self.images_train_lbl = QLabel("Images in your training dataset: ", self)
        self.images_train_value_lbl = QLabel("", self)
        self.images_val_lbl = QLabel("Images in your validation dataset: ", self)
        self.images_val_value_lbl = QLabel("", self)
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
        self.warm_up_lbl = QLabel("Warm up iterations: ", self)
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
        self.warm_up_value = QTextEdit(self)

        self.epoch_progress = QProgressBar(self)

        self.time_box = QGroupBox("Time of your training", self)
        self.epoch_time_lbl = QLabel("Time of your last epoch: ", self)
        self.epoch_time_value = QLabel("0.00 s", self)
        self.val_time_lbl = QLabel("Time of your last evaluation: ", self)
        self.val_time_value = QLabel("0.00 s", self)

        self.best_model_box = QGroupBox("Your best model so far", self)
        self.after_epoch_lbl = QLabel("Score form epoch: ", self)
        self.after_epoch_value = QLabel("0", self)
        self.f1_box_lbl = QLabel("Box F score: ", self)
        self.f1_box_value = QLabel("0.00", self)
        self.f1_mask_lbl = QLabel("Mask F score: ", self)
        self.f1_mask_value = QLabel("0.00", self)

        self.total_training_time_lbl = QLabel("Your total training time: ", self)
        self.total_training_time_value = QLabel("...", self)

        self.set_gui()
        self.set_second_stage()

    def change_stage(self, stage=True):

        self.epoch_progress.setVisible(stage)
        self.total_training_time_lbl.setVisible(stage)
        self.total_training_time_value.setVisible(stage)
        self.time_box.setVisible(stage)
        self.best_model_box.setVisible(stage)
        self.data_params_box.setVisible(stage)

        self.start_training_btn.setDisabled(stage)
        self.validating_btn.setVisible(not stage)
        self.training_params_box.setVisible(not stage)
        self.start_fresh_box.setVisible(not stage)
        self.use_default_params_btn.setVisible(not stage)
        self.clear_btn.setVisible(not stage)

    def set_second_stage(self):
        self.epoch_progress.setGeometry(self.starting_lbl_x + 150, self.starting_lbl_y + 50, 684, 50)
        self.total_training_time_lbl.setGeometry(self.starting_lbl_x + 40, 400, 260, 40)
        self.total_training_time_value.setGeometry(self.starting_lbl_x + 300, 400, 300, 40)
        self.total_training_time_lbl.setFont(QFont('Arial', 12))
        self.total_training_time_value.setFont(QFont('Arial', 12))
        self.epoch_progress.setValue(0)

        self.set_time_section()
        self.set_best_model_section()
        self.set_dataset_section()

    def set_gui(self):
        self.set_config_section()
        self.set_header_section()
        self.set_additional_settings()

        self.title.setWordWrap(True)
        self.title.setFont(QFont('Arial', 40))
        self.title.setGeometry(200, 10, 400, 80)
        self.title.setStyleSheet("font-style: italic;\n"
                                 "font-weight: bold;")
        self.change_stage(False)
        self.start_training_btn.setDisabled(True)

    def set_time_section(self):
        gbox = QGridLayout()
        self.time_box.setGeometry(self.starting_lbl_x + 362, 210, 260, 130)

        self.epoch_time_lbl.setWordWrap(True)
        self.epoch_time_lbl.setFont(QFont('Arial', 10))
        self.epoch_time_lbl.setStyleSheet("color: Gray")

        self.epoch_time_value.setWordWrap(True)
        self.epoch_time_value.setFont(QFont('Arial', 10))
        self.epoch_time_value.setStyleSheet("color: Gray")

        self.val_time_lbl.setWordWrap(True)
        self.val_time_lbl.setFont(QFont('Arial', 10))
        self.val_time_lbl.setStyleSheet("color: Gray")

        self.val_time_value.setWordWrap(True)
        self.val_time_value.setFont(QFont('Arial', 10))
        self.val_time_value.setStyleSheet("color: Gray")

        gbox.addWidget(self.epoch_time_lbl, 0, 0)
        gbox.addWidget(self.epoch_time_value, 0, 1)
        gbox.addWidget(self.val_time_lbl, 1, 0)
        gbox.addWidget(self.val_time_value, 1, 1)
        self.time_box.setLayout(gbox)

    def set_best_model_section(self):
        gbox = QGridLayout()
        self.best_model_box.setGeometry(self.starting_lbl_x + 40, 210, 260, 130)
        self.after_epoch_lbl.setWordWrap(True)
        self.after_epoch_lbl.setFont(QFont('Arial', 10))
        self.after_epoch_lbl.setStyleSheet("color: Gray")

        self.after_epoch_value.setWordWrap(True)
        self.after_epoch_value.setFont(QFont('Arial', 10))
        self.after_epoch_value.setStyleSheet("color: Gray")

        self.f1_box_lbl.setWordWrap(True)
        self.f1_box_lbl.setFont(QFont('Arial', 10))
        self.f1_box_lbl.setStyleSheet("color: Gray")

        self.f1_box_value.setWordWrap(True)
        self.f1_box_value.setFont(QFont('Arial', 10))
        self.f1_box_value.setStyleSheet("color: Gray")

        self.f1_mask_lbl.setWordWrap(True)
        self.f1_mask_lbl.setFont(QFont('Arial', 10))
        self.f1_mask_lbl.setStyleSheet("color: Gray")

        self.f1_mask_value.setWordWrap(True)
        self.f1_mask_value.setFont(QFont('Arial', 10))
        self.f1_mask_value.setStyleSheet("color: Gray")

        gbox.addWidget(self.after_epoch_lbl, 0, 0)
        gbox.addWidget(self.after_epoch_value, 0, 1)
        gbox.addWidget(self.f1_box_lbl, 1, 0)
        gbox.addWidget(self.f1_box_value, 1, 1)
        gbox.addWidget(self.f1_mask_lbl, 2, 0)
        gbox.addWidget(self.f1_mask_value, 2, 1)
        self.best_model_box.setLayout(gbox)

    def set_dataset_section(self):
        gbox = QGridLayout()
        self.data_params_box.setGeometry(self.starting_lbl_x + 362*2 - 40, 210, 260, 130)

        self.images_train_lbl.setWordWrap(True)
        self.images_train_lbl.setFont(QFont('Arial', 10))
        self.images_train_lbl.setStyleSheet("color: Gray")

        self.images_train_value_lbl.setWordWrap(True)
        self.images_train_value_lbl.setFont(QFont('Arial', 10))
        self.images_train_value_lbl.setStyleSheet("color: Gray")

        self.images_val_lbl.setWordWrap(True)
        self.images_val_lbl.setFont(QFont('Arial', 10))
        self.images_val_lbl.setStyleSheet("color: Gray")

        self.images_val_value_lbl.setWordWrap(True)
        self.images_val_value_lbl.setFont(QFont('Arial', 10))
        self.images_val_value_lbl.setStyleSheet("color: Gray")

        gbox.addWidget(self.images_train_lbl, 0, 0)
        gbox.addWidget(self.images_train_value_lbl, 0, 1)
        gbox.addWidget(self.images_val_lbl, 1, 0)
        gbox.addWidget(self.images_val_value_lbl, 1, 1)

        self.data_params_box.setLayout(gbox)

    def set_header_section(self):
        self.back_to_menu_btn.setGeometry(20, 20, 100, self.lbl_height + 10)

        self.start_training_btn.setGeometry(300, 650, 424, self.lbl_height + 20)
        self.start_training_btn.setFont(QFont('Arial', 15))
        self.start_training_btn.setStyleSheet("font-weight: bold;")
        self.start_training_btn.clicked.connect(self.process_training)

        self.validating_btn.setGeometry(704, 400, 300, self.lbl_height + 10)
        self.validating_btn.setFont(QFont('Arial', 10))
        self.validating_btn.setStyleSheet("font-weight: bold;")
        self.validating_btn.clicked.connect(self.validate_params)

    def set_additional_settings(self):
        self.start_fresh_box.addItems(["Start fresh", "Continue training"])
        self.start_fresh_box.setGeometry(self.starting_lbl_x, 400, 160, self.lbl_height + 10)

        self.clear_btn.setGeometry(self.starting_lbl_x + 180, 400, 160, self.lbl_height + 10)
        self.clear_btn.clicked.connect(self.clear_params)
        self.use_default_params_btn.setGeometry(self.starting_lbl_x + 360, 400, 160, self.lbl_height + 10)
        self.use_default_params_btn.clicked.connect(self.set_default_params)

    def set_config_section(self):
        gbox = QGridLayout()

        self.warning_lbl.setWordWrap(True)
        self.warning_lbl.setFont(QFont('Arial', 10))
        self.warning_lbl.setStyleSheet("color: red")

        self.device_lbl.setWordWrap(True)
        self.device_lbl.setFont(QFont('Arial', 10))
        self.learning_rate_lbl.setWordWrap(True)
        self.learning_rate_lbl.setFont(QFont('Arial', 10))
        self.learning_steps_lbl.setWordWrap(True)
        self.learning_steps_lbl.setFont(QFont('Arial', 10))
        self.decay_lbl.setWordWrap(True)
        self.decay_lbl.setFont(QFont('Arial', 10))
        self.momentum_lbl.setWordWrap(True)
        self.momentum_lbl.setFont(QFont('Arial', 10))
        self.num_epoch_lbl.setWordWrap(True)
        self.num_epoch_lbl.setFont(QFont('Arial', 10))
        self.iterations_lbl.setWordWrap(True)
        self.iterations_lbl.setFont(QFont('Arial', 10))
        self.dir_path_lbl.setWordWrap(True)
        self.dir_path_lbl.setFont(QFont('Arial', 10))
        self.class_num_lbl.setWordWrap(True)
        self.class_num_lbl.setFont(QFont('Arial', 10))
        self.seed_lbl.setWordWrap(True)
        self.seed_lbl.setFont(QFont('Arial', 10))
        self.warm_up_lbl.setWordWrap(True)
        self.warm_up_lbl.setFont(QFont('Arial', 10))

        self.device_value.addItems(["cpu", "cuda"])
        self.brows_btn.clicked.connect(self.get_path)
        self.path_content.setDisabled(True)

        gbox.addWidget(self.device_lbl, 0, 0, 1, 2)
        gbox.addWidget(self.device_value, 0, 2, 1, 1)
        gbox.addWidget(self.learning_rate_lbl, 1, 0, 1, 2)
        gbox.addWidget(self.learning_rate_value, 1, 2, 1, 1)
        gbox.addWidget(self.learning_steps_lbl, 2, 0, 1, 2)
        gbox.addWidget(self.learning_steps_value, 2, 2, 1, 1)
        gbox.addWidget(self.decay_lbl, 3, 0, 1, 2)
        gbox.addWidget(self.decay_value, 3, 2, 1, 1)
        gbox.addWidget(self.momentum_lbl, 4, 0, 1, 2)
        gbox.addWidget(self.momentum_value, 4, 2, 1, 1)

        gbox.addWidget(self.num_epoch_lbl, 0, 3, 1, 2)
        gbox.addWidget(self.epoch_value, 0, 5, 1, 1)
        gbox.addWidget(self.iterations_lbl, 1, 3, 1, 2)
        gbox.addWidget(self.iterations_value, 1, 5, 1, 1)
        gbox.addWidget(self.warm_up_lbl, 2, 3, 1, 2)
        gbox.addWidget(self.warm_up_value, 2, 5, 1, 1)
        gbox.addWidget(self.class_num_lbl, 3, 3, 1, 2)
        gbox.addWidget(self.class_value, 3, 5, 1, 1)
        gbox.addWidget(self.seed_lbl, 4, 3, 1, 2)
        gbox.addWidget(self.seed_value, 4, 5, 1, 1)

        gbox.addWidget(self.dir_path_lbl, 5, 0, 1, 2)
        gbox.addWidget(self.brows_btn, 5, 3, 1, 1)
        gbox.addWidget(self.path_content, 5, 2, 1, 1)
        gbox.addWidget(self.warning_lbl, 5, 5, 1, 1)

        self.training_params_box.setGeometry(self.starting_lbl_x, self.starting_lbl_y, 984, 300)
        self.training_params_box.setLayout(gbox)

    def set_default_params(self):
        cfg = Configuration()
        self.learning_rate_value.setText(str(cfg.learning_rate))
        self.learning_steps_value.setText("200 400")
        self.decay_value.setText(str(cfg.decay))
        self.momentum_value.setText(str(cfg.momentum))
        self.epoch_value.setText(str(cfg.number_of_epochs))
        self.iterations_value.setText(str(cfg.number_of_iterations))
        self.path_content.setText('Dataset')
        self.dir_path = 'Dataset'
        self.class_value.setText(str(cfg.number_of_classes))
        self.seed_value.setText(str(cfg.seed))
        self.warm_up_value.setText(str(cfg.iterations_to_warmup))

    def clear_params(self):
        self.learning_rate_value.setText("")
        self.learning_steps_value.setText("")
        self.decay_value.setText("")
        self.momentum_value.setText("")
        self.epoch_value.setText("")
        self.iterations_value.setText("")
        self.path_content.setText("")
        self.class_value.setText("")
        self.seed_value.setText("")
        self.warm_up_value.setText("")

    def get_path(self):
        self.dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory", "C:\\")
        self.path_content.setText(self.dir_path)

    def get_seed(self):
        try:
            return int(self.seed_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_epoch(self):
        try:
            return int(self.epoch_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_classes(self):
        try:
            return int(self.class_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_iterations(self):
        try:
            return int(self.iterations_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_momentum(self):
        try:
            return float(self.momentum_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_decay(self):
        try:
            return float(self.decay_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_learning_rate(self):
        try:
            return float(self.learning_rate_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_learning_steps(self):
        try:
            lr_steps = list(self.learning_steps_value.toPlainText().split(" "))
            lr_steps = [int(x) for x in lr_steps]
            return lr_steps
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def get_device(self):
        return str(self.device_value.currentText())

    def get_iterations_to_warmup(self):
        try:
            return int(self.warm_up_value.toPlainText())
        except:
            self.warning_lbl.setText("INCORECT PARAMETERS")
            return -1

    def update_progress_bar(self):
        value = self.present_epoch / self.parameters['number_of_epochs']
        value = int(value * 100)
        self.epoch_progress.setValue(value)

    def print_output(self, s):
        print(s)

    def training_complete(self):
        print("TRAINING COMPLETE!")

    def process_training(self):
        self.change_stage(True)
        worker = Worker(self.train)
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.training_complete)

        self.threadpool.start(worker)

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
            "publishing_losses_frequency": 20,
            "checkpoint_path": 'ckpt/',
            "learning_rate_lambda": 0.1,
            "model_path": 'model/',
            "iterations_to_warmup": self.get_iterations_to_warmup(),
            "result_path": 'ckpt/result'
        }
        return params

    def validate_params(self):
        params = self.get_all_params()
        wrong_params = False
        for p in params.values():
            if p == -1:
                wrong_params = True
        if wrong_params:
            self.start_training_btn.setDisabled(True)
        else:
            self.warning_lbl.setText("")
            self.start_training_btn.setDisabled(False)
            self.parameters = params

    def train(self):
        device = torch.device(self.parameters['device'])

        if str(self.start_fresh_box.currentText()) == "Start fresh":
            prefix, ext = os.path.splitext(self.parameters['checkpoint_path'])
            files = glob.glob(prefix + "-*" + ext)
            for i in range(len(files)):
                os.remove("{}".format(files[i]))

        try:
            train_set = algorithm.COCODataset(self.parameters['dataset_dir'], "Train", train=True)
            indices = torch.randperm(len(train_set)).tolist()
            train_set = torch.utils.data.Subset(train_set, indices)

            val_set = algorithm.COCODataset(self.parameters['dataset_dir'], "Validation", train=True)
            self.images_train_value_lbl.setText('{}'.format(len(train_set)))
            self.images_val_value_lbl.setText('{}'.format(len(val_set)))
            QApplication.processEvents()

            model = algorithm.resnet50_for_mask_rcnn(True, self.parameters['number_of_classes']).to(device)
        except Exception as e:
            print(e)
            self.change_stage(False)
            return

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

        for epoch in range(starting_epoch, self.parameters['number_of_epochs']):
            self.present_epoch = epoch
            self.update_progress_bar()

            training_epoch_time = time.time()
            self.parameters['learning_epoch'] = decrease(epoch) * self.parameters['learning_rate']

            try:
                algorithm.train_epoch(model, optimizer, train_set, device, epoch, self.parameters)
                training_epoch_time = time.time() - training_epoch_time
                if training_epoch_time > 60:
                    training_epoch_time_str = training_epoch_time/60
                    training_epoch_time_str = format(training_epoch_time_str, ".2f")
                    training_epoch_time_str += ' min'
                else:
                    training_epoch_time_str = format(training_epoch_time, ".2f")
                    training_epoch_time_str += ' s'

                self.epoch_time_value.setText(training_epoch_time_str)

                validation_epoch_time = time.time()
                eval_output = algorithm.evaluate(model, val_set, device, self.parameters)
                validation_epoch_time = time.time() - validation_epoch_time

                print(eval_output)

                if validation_epoch_time > 60:
                    validation_epoch_time_str = validation_epoch_time/60
                    validation_epoch_time_str = format(validation_epoch_time_str, ".2f")
                    validation_epoch_time_str += ' min'
                else:
                    validation_epoch_time_str = format(validation_epoch_time, ".2f")
                    validation_epoch_time_str += ' s'

                self.val_time_value.setText(validation_epoch_time_str)
                QApplication.processEvents()
            except Exception as e:
                print(e)
                self.change_stage(False)
                return

            trained_epoch = epoch + 1
            maskF = eval_output.get_AF()
            if maskF['mask FScore'] > self.best_model_by_maskF:
                self.best_model_by_maskF = maskF['mask FScore']
                algorithm.save_best(model, optimizer, trained_epoch,
                                    self.parameters['model_path'], eval_info=str(eval_output))
                self.f1_box_value.setText('{}'.format(maskF['bbox FScore']))
                self.f1_mask_value.setText('{}'.format(maskF['mask FScore']))
                self.after_epoch_value.setText('{}'.format(epoch + 1))
                QApplication.processEvents()

            algorithm.save_checkpoint(model, optimizer, trained_epoch,
                                      self.parameters['checkpoint_path'], eval_info=str(eval_output))

            prefix, ext = os.path.splitext(self.parameters['checkpoint_path'])
            checkpoints = glob.glob(prefix + "-*" + ext)
            checkpoints.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
            n = 3
            if len(checkpoints) > n:
                for i in range(len(checkpoints) - n):
                    os.remove("{}".format(checkpoints[i]))

        self.present_epoch = self.present_epoch + 1
        self.update_progress_bar()

        total_training_time = time.time() - since
        self.total_training_time_value.setText('{} s'.format(format(total_training_time, ".2f")))
        QApplication.processEvents()
        return "Done."

