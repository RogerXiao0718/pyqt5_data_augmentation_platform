import math
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QStringListModel
from UI import Ui_MainWindow
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

    # 負責綁定事件
    def event_binding(self):
        self.ui.filename_listwidget.clicked.connect(self.listViewClicked)
        self.ui.action_loadImage.triggered.connect(self.loadFilesClicked)
        self.ui.action_loadFolder.triggered.connect(self.loadFolderClicked)

    def loadFilesClicked(self):
        loaded_filepaths, filetype = QFileDialog.getOpenFileNames(self, "Open Images", "./",
                                                                  filter="Images (*.png *.jpg *.jiff *.bmp)")
        if loaded_filepaths:
            self.data["loaded_filepaths"] = loaded_filepaths
            self.data["loaded_filenames"] = list(map(lambda filepath: os.path.basename(filepath), loaded_filepaths))
            self.updateFilenameList(self.data["loaded_filenames"])

    def loadFolderClicked(self):
        loaded_folder = QFileDialog.getExistingDirectory(self, "Open Directory", "./")
        extension_whitelist = [".png", ".jpg", ".jiff", ".bmp"]
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

    def listViewClicked(self, qmodelIndex):
        QMessageBox.information(self, "Prompt", f"You selected {self.data['loaded_filenames'][qmodelIndex.row()]}")
