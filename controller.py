import math
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QStringListModel
from UI import Ui_MainWindow
import numpy as np
import random
import os
import cv2

random.seed(7414)


class MainWindow_controller(QtWidgets.QMainWindow):
    ROISelectionActivated = False

    # 建構子
    def __init__(self):
        super(MainWindow_controller, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.event_binding()

    # 設定元件
    def setup_control(self):
        self.data = {}
        self.ui.statusLabel = QtWidgets.QLabel(self.statusBar())
        self.ui.statusLabel.setText("")
        self.statusBar().addPermanentWidget(self.ui.statusLabel)

    # 負責綁定事件
    def event_binding(self):
        self.ui.filename_listwidget.clicked.connect(self.listViewClicked)
        self.ui.action_loadImage.triggered.connect(self.loadFilesClicked)
        self.ui.action_loadFolder.triggered.connect(self.loadFolderClicked)

    def loadFilesClicked(self):
        loaded_filepaths, filetype = QFileDialog.getOpenFileNames(self, "Open Images", "./",
                                                                  filter="Images (*.png *.jpg *.jfif *.bmp)")
        if loaded_filepaths:
            self.data["loaded_filepaths"] = loaded_filepaths
            self.data["loaded_filenames"] = list(map(lambda filepath: os.path.basename(filepath), loaded_filepaths))
            self.updateFilenameList(self.data["loaded_filenames"])

    def loadFolderClicked(self):
        loaded_folder = QFileDialog.getExistingDirectory(self, "Open Directory", "./")
        extension_whitelist = [".png", ".jpg", ".jfif", ".bmp"]
        if loaded_folder:
            loaded_filenames = os.listdir(loaded_folder)
            loaded_filenames = list(filter(lambda filename: os.path.splitext(filename)[1] in extension_whitelist, loaded_filenames))
            loaded_filepaths = list(map(lambda filename: os.path.join(loaded_folder, filename), loaded_filenames))
            self.data["loaded_filenames"] = loaded_filenames
            self.data["loaded_filepaths"] = loaded_filepaths
            self.updateFilenameList(self.data["loaded_filenames"])

    def updateFilenameList(self, loaded_filenames):
        self.ui.filename_listwidget.clear()
        self.ui.filename_listwidget.addItems(
            [f"[{i}] {filename}" for i, filename in enumerate(loaded_filenames)])

    def imageDisplay(self, image):
        self.resize_displayed_images(image)

        if len(self.displayed_image.shape) == 3:
            height, width, channel = self.displayed_image.shape
            bytesPerLine = width * channel
            self.qimage = QImage(self.displayed_image, width, height, bytesPerLine, QImage.Format_BGR888)
            self.ui.imageDisplayer.setPixmap(QPixmap.fromImage(self.qimage))
        elif len(self.displayed_image.shape) == 2:
            height, width = self.displayed_image.shape
            bytesPerLine = width * 1
            self.qimage = QImage(self.displayed_image, width, height, bytesPerLine, QImage.Format_Grayscale8)
            self.ui.imageDisplayer.setPixmap(QPixmap.fromImage(self.qimage))

    def resize_displayed_images(self, image):
        displayer_size = self.ui.imageDisplayer.size()
        image_height = image.shape[0]
        image_width = image.shape[1]
        if image_height > displayer_size.height() and image_height >= image_width:
            image_hw_ratio = image_height / image_width
            new_image_height = displayer_size.height()
            new_image_width = int(new_image_height / image_hw_ratio)
            self.displayed_image = cv2.resize(image, (new_image_width, new_image_height),
                                              interpolation=cv2.INTER_AREA)
        elif image_width > displayer_size.width() and image_width >= image_height:
            image_wh_ratio = image_width / image_height
            new_image_width = displayer_size.width()
            new_image_height = int(new_image_width / image_wh_ratio)
            self.displayed_image = cv2.resize(image, (new_image_width, new_image_height),
                                              interpolation=cv2.INTER_AREA)
        else:
            self.displayed_image = image

    def listViewClicked(self, qmodelIndex):
        self.cv2_image = cv2.imdecode(np.fromfile(self.data["loaded_filepaths"][qmodelIndex.row()], dtype=np.uint8), -1)
        self.imageDisplay(self.cv2_image)
        self.ui.statusLabel.setText(self.data["loaded_filenames"][qmodelIndex.row()])


