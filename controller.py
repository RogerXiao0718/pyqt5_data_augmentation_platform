import math
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QStringListModel
from utils.brightness_contrast_augmentation import dataAugmentation_contrast_brightness as brightness_contrast_augmentation
from utils.affine_transform_augmentation import coco_data_augmentation as affine_transform_augmentation
from UI import Ui_MainWindow
import numpy as np
import random
import os
import cv2
import threading

random.seed(9712)


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
        self.ui.action_loadAnnotation.triggered.connect(self.loadAnnotationPathClicked)
        self.ui.action_selectSavePath.triggered.connect(self.selectSavePathClicked)
        self.ui.rotationSlider.valueChanged.connect(self.onRotationSliderChanged)
        self.ui.brightnessSlider.valueChanged.connect(self.onBrightnessSliderChanged)
        self.ui.contrastSlider.valueChanged.connect(self.onContrastSliderChanged)
        self.ui.saturateSlider.valueChanged.connect(self.onSaturateSliderChanged)
        self.ui.hueSlider.valueChanged.connect(self.onHueSliderChanged)
        self.ui.augmentationButton.clicked.connect(self.augmentationButtonClicked)

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

    def loadAnnotationPathClicked(self):
        loaded_annotationPath, filetype = QFileDialog.getOpenFileName(self, "Open Images", "./",
                                                                  filter="Annotation (*.json)")
        if loaded_annotationPath:
            self.data["loaded_annotationPath"] = loaded_annotationPath


    def selectSavePathClicked(self):
        augmentation_savePath = QFileDialog.getExistingDirectory(self, "Save Directory", "./")
        if augmentation_savePath:
            self.data["augmentation_savePath"] = augmentation_savePath
            self.ui.statusLabel.setText(f"儲存位置: {augmentation_savePath}")

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

    def onRotationSliderChanged(self):
        self.ui.rotationValueLabel.setText(str(self.ui.rotationSlider.value()))

    def onBrightnessSliderChanged(self):
        self.ui.brightnessValueLabel.setText(str(self.ui.brightnessSlider.value()))

    def onContrastSliderChanged(self):
        self.ui.contrastValueLabel.setText(str(self.ui.contrastSlider.value()))

    def onSaturateSliderChanged(self):
        self.ui.saturateValueLabel.setText(str(self.ui.saturateSlider.value()))

    def onHueSliderChanged(self):
        self.ui.hueValueLabel.setText(str(self.ui.hueSlider.value()))

    def augmentationButtonClicked(self):
        affine_transform_rotation_value = int(self.ui.rotationSlider.value())
        affine_transform_rotation_range = (-affine_transform_rotation_value, affine_transform_rotation_value + 1)
        affine_transform_aug_times = int(self.ui.augmentation_inputBox.text())
        affine_transform_angle_step = int(affine_transform_rotation_value * 2 / affine_transform_aug_times)
        affine_transform_style = [] #[[angle, flip, histo]]
        imageColoredAugmentationCachePath = os.path.join(self.data["augmentation_savePath"], "./cache")
        if not os.path.exists(imageColoredAugmentationCachePath):
            os.mkdir(imageColoredAugmentationCachePath)
        finalImageSavePath = os.path.join(self.data["augmentation_savePath"], './images')
        if not os.path.exists(finalImageSavePath):
            os.mkdir(finalImageSavePath)
        isAnnotationAugment = "loaded_annotationPath" in self.data
        annotationSavePath = os.path.join(self.data["augmentation_savePath"],
                                          "./augmented_annotation.json") if isAnnotationAugment else None
        annotation_path = self.data["loaded_annotationPath"] if "loaded_annotationPath" in self.data else None



        brightnessValue = self.ui.brightnessSlider.value() / 10
        contrastValue = self.ui.contrastSlider.value() / 10
        saturateValue = self.ui.saturateSlider.value() / 10
        hueValue = self.ui.hueSlider.value() / 10
        isHighLight = self.ui.highlight_checkBox.isChecked()
        coloredAugData = {
            "brightness": brightnessValue,
            "contrast": contrastValue,
            "saturation": saturateValue,
            "hue": hueValue,
            "highlight": isHighLight
        }

        for rotation_angle in range(affine_transform_rotation_range[0], affine_transform_rotation_range[1],
                                    affine_transform_angle_step):
            affine_transform_style.append([rotation_angle, self.ui.flip_checkBox.isChecked(), self.ui.histo_checkBox.isChecked()])
        def augmentationJob():
            brightness_contrast_augmentation(self.data["loaded_filepaths"], imageColoredAugmentationCachePath,
                                             coloredAugData, logLabel=self.ui.statusLabel)

            affineTransformImageFilenames = os.listdir(imageColoredAugmentationCachePath)
            affineTransformImagePaths = list(map(lambda filename: os.path.join(imageColoredAugmentationCachePath, filename), affineTransformImageFilenames))

            affine_transform_augmentation(affineTransformImagePaths, annotation_path, finalImageSavePath, styles=affine_transform_style,
                                          json_save_path=annotationSavePath, json_Augmentation=isAnnotationAugment, save_image=True,
                                          logLabel = self.ui.statusLabel)

            self.ui.statusLabel.setText("擴增完成")
            # delete cache
            self.deleteCache(imageColoredAugmentationCachePath)

        augmentationThread = threading.Thread(target=augmentationJob)
        augmentationThread.start()


    def deleteCache(self, imageAffineAugmentationCachePath):
        for delete_filename in os.listdir(imageAffineAugmentationCachePath):
            os.remove(os.path.join(imageAffineAugmentationCachePath, delete_filename))
        os.rmdir(imageAffineAugmentationCachePath)



